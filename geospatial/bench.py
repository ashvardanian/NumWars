#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geospatial distance benchmarks: NumKong vs Python alternatives.

Benchmarks batched coordinate-pair distances for:
- haversine
- vincenty-like geodesic distance

Can be run with uv:
    uv run --with numkong,numpy,geopy,tabulate geospatial/bench.py

Or with traditional pip:
    pip install -e ".[geospatial]"
    python geospatial/bench.py
"""

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tabulate
from geopy.distance import geodesic, great_circle

try:
    import numkong as nk
except ImportError:
    print("Error: numkong not found. Install with: pip install numkong")
    sys.exit(1)

try:
    from utils import (
        add_common_args,
        calculate_mps,
        measure_average_duration,
        normalize_dtype_name,
        print_results_table,
        should_run_benchmark,
    )
except ImportError:
    print("Error: Could not import utils.py. Make sure it's in the parent directory.")
    sys.exit(1)


EARTH_RADIUS_KM = 6371.0


@dataclass
class BenchmarkResult:
    workload: str
    library: str
    input_dtype: str
    output_dtype: str
    count: int
    duration_secs: float
    primary_value: float


def build_coords(
    count: int, seed: int, dtype=np.float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a_lats = rng.uniform(-90.0, 90.0, size=count).astype(dtype)
    a_lons = rng.uniform(-180.0, 180.0, size=count).astype(dtype)
    b_lats = rng.uniform(-90.0, 90.0, size=count).astype(dtype)
    b_lons = rng.uniform(-180.0, 180.0, size=count).astype(dtype)
    return a_lats, a_lons, b_lats, b_lons


def baseline_haversine(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> float:
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = math.radians(latitude_b - latitude_a)
    delta_lon = math.radians(longitude_b - longitude_a)
    half_chord_squared = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * math.asin(math.sqrt(half_chord_squared))


def baseline_vincenty(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a = 6_378_137.0
    f = 1.0 / 298.257_223_563
    b = a * (1.0 - f)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    l = math.radians(lon2 - lon1)

    u1 = math.atan((1.0 - f) * math.tan(phi1))
    u2 = math.atan((1.0 - f) * math.tan(phi2))
    sin_u1 = math.sin(u1)
    cos_u1 = math.cos(u1)
    sin_u2 = math.sin(u2)
    cos_u2 = math.cos(u2)

    lam = l
    sin_sigma = 0.0
    cos_sigma = 0.0
    sigma = 0.0
    sin_alpha = 0.0
    cos_sq_alpha = 0.0
    cos_2sigma_m = 0.0

    for _ in range(200):
        sin_lambda = math.sin(lam)
        cos_lambda = math.cos(lam)
        sin_sigma = math.sqrt(
            (cos_u2 * sin_lambda) ** 2
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda) ** 2
        )
        if sin_sigma == 0.0:
            return 0.0

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha
        cos_2sigma_m = (
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
            if cos_sq_alpha != 0.0
            else 0.0
        )

        c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha))
        lam_prev = lam
        lam = l + (1.0 - c) * f * sin_alpha * (
            sigma
            + c
            * sin_sigma
            * (
                cos_2sigma_m
                + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
            )
        )
        if abs(lam - lam_prev) < 1e-12:
            break

    u_sq = cos_sq_alpha * (a * a - b * b) / (b * b)
    cap_a = 1.0 + u_sq / 16384.0 * (
        4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq))
    )
    cap_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))
    delta_sigma = (
        cap_b
        * sin_sigma
        * (
            cos_2sigma_m
            + cap_b
            / 4.0
            * (
                cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                - cap_b
                / 6.0
                * cos_2sigma_m
                * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)
            )
        )
    )
    return b * cap_a * (sigma - delta_sigma) / 1000.0


def benchmark_numkong(
    workload: str,
    func,
    a_lats: np.ndarray,
    a_lons: np.ndarray,
    b_lats: np.ndarray,
    b_lons: np.ndarray,
    warmup: float,
    profile: float,
) -> BenchmarkResult:
    input_dtype = normalize_dtype_name(a_lats.dtype)
    sample_output = func(a_lats, a_lons, b_lats, b_lons)
    output_dtype = normalize_dtype_name(getattr(sample_output, "dtype", a_lats.dtype))
    out = np.empty(a_lats.shape[0], dtype=a_lats.dtype)
    duration = measure_average_duration(
        lambda: func(a_lats, a_lons, b_lats, b_lons, out=out),
        warmup,
        profile,
    )
    return BenchmarkResult(
        workload=workload,
        library="NumKong",
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        count=a_lats.shape[0],
        duration_secs=duration,
        primary_value=calculate_mps(a_lats.shape[0], duration),
    )


def benchmark_serial(
    workload: str,
    func,
    a_lats: np.ndarray,
    a_lons: np.ndarray,
    b_lats: np.ndarray,
    b_lons: np.ndarray,
    warmup: float,
    profile: float,
) -> BenchmarkResult:
    out = np.empty(a_lats.shape[0], dtype=np.float64)

    def run_once() -> None:
        for index in range(a_lats.shape[0]):
            out[index] = func(
                float(a_lats[index]),
                float(a_lons[index]),
                float(b_lats[index]),
                float(b_lons[index]),
            )

    duration = measure_average_duration(run_once, warmup, profile)
    return BenchmarkResult(
        workload=workload,
        library="Serial",
        input_dtype=normalize_dtype_name(a_lats.dtype),
        output_dtype="f64",
        count=a_lats.shape[0],
        duration_secs=duration,
        primary_value=calculate_mps(a_lats.shape[0], duration),
    )


def benchmark_geopy(
    workload: str,
    func,
    a_lats: np.ndarray,
    a_lons: np.ndarray,
    b_lats: np.ndarray,
    b_lons: np.ndarray,
    warmup: float,
    profile: float,
) -> BenchmarkResult:
    pairs = list(
        zip(
            zip(a_lats.tolist(), a_lons.tolist()), zip(b_lats.tolist(), b_lons.tolist())
        )
    )
    out = np.empty(len(pairs), dtype=np.float64)

    def run_once() -> None:
        for index, (point_a, point_b) in enumerate(pairs):
            out[index] = func(point_a, point_b).km

    duration = measure_average_duration(run_once, warmup, profile)
    return BenchmarkResult(
        workload=workload,
        library="geopy",
        input_dtype=normalize_dtype_name(a_lats.dtype),
        output_dtype="f64",
        count=len(pairs),
        duration_secs=duration,
        primary_value=calculate_mps(len(pairs), duration),
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "geospatial",
        "workload": result.workload,
        "benchmark_id": f"geospatial/{result.workload}/{result.library.lower()}/{result.input_dtype}",
        "library": result.library,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_signature": f"{result.input_dtype} \u2192 {result.output_dtype}",
        "count": result.count,
        "primary_value": result.primary_value,
        "unit": "MP/s",
        "duration_secs": result.duration_secs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark geospatial distance kernels"
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Choose human-readable table output or machine-readable JSON (default: table).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=int(os.environ.get("NUMWARS_DIMS", "2048")),
        help="Number of coordinate pairs to benchmark (default: NUMWARS_DIMS or 2048).",
    )
    args = parser.parse_args()

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as exc:
            print(f"Invalid filter regex: {exc}", file=sys.stderr)
            sys.exit(2)

    a_lats, a_lons, b_lats, b_lons = build_coords(args.count, args.seed)
    a_lats64, a_lons64, b_lats64, b_lons64 = build_coords(
        args.count, args.seed, dtype=np.float64
    )
    benchmarks = [
        (
            "geospatial/haversine/numkong/f32",
            lambda: benchmark_numkong(
                "haversine",
                nk.haversine,
                a_lats,
                a_lons,
                b_lats,
                b_lons,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/haversine/numkong/f64",
            lambda: benchmark_numkong(
                "haversine",
                nk.haversine,
                a_lats64,
                a_lons64,
                b_lats64,
                b_lons64,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/haversine/serial/f32",
            lambda: benchmark_serial(
                "haversine",
                baseline_haversine,
                a_lats,
                a_lons,
                b_lats,
                b_lons,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/haversine/serial/f64",
            lambda: benchmark_serial(
                "haversine",
                baseline_haversine,
                a_lats64,
                a_lons64,
                b_lats64,
                b_lons64,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/haversine/geopy/f32",
            lambda: benchmark_geopy(
                "haversine",
                great_circle,
                a_lats,
                a_lons,
                b_lats,
                b_lons,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/haversine/geopy/f64",
            lambda: benchmark_geopy(
                "haversine",
                great_circle,
                a_lats64,
                a_lons64,
                b_lats64,
                b_lons64,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/vincenty/numkong/f32",
            lambda: benchmark_numkong(
                "vincenty",
                nk.vincenty,
                a_lats,
                a_lons,
                b_lats,
                b_lons,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/vincenty/numkong/f64",
            lambda: benchmark_numkong(
                "vincenty",
                nk.vincenty,
                a_lats64,
                a_lons64,
                b_lats64,
                b_lons64,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/vincenty/serial/f32",
            lambda: benchmark_serial(
                "vincenty",
                baseline_vincenty,
                a_lats,
                a_lons,
                b_lats,
                b_lons,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/vincenty/serial/f64",
            lambda: benchmark_serial(
                "vincenty",
                baseline_vincenty,
                a_lats64,
                a_lons64,
                b_lats64,
                b_lons64,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/vincenty/geopy/f32",
            lambda: benchmark_geopy(
                "vincenty",
                geodesic,
                a_lats,
                a_lons,
                b_lats,
                b_lons,
                args.warmup,
                args.time_limit,
            ),
        ),
        (
            "geospatial/vincenty/geopy/f64",
            lambda: benchmark_geopy(
                "vincenty",
                geodesic,
                a_lats64,
                a_lons64,
                b_lats64,
                b_lons64,
                args.warmup,
                args.time_limit,
            ),
        ),
    ]

    results = []
    for benchmark_id, benchmark_fn in benchmarks:
        if not should_run_benchmark(benchmark_id, filter_pattern):
            continue
        results.append(result_to_entry(benchmark_fn()))

    if args.output_format == "json":
        json.dump({"results": results}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if not results:
        print("No benchmarks matched the filter.")
        return

    table_rows = [
        {
            "benchmark": entry["benchmark_id"],
            "library": entry["library"],
            "signature": entry["display_signature"],
            "duration": f"{entry['duration_secs'] * 1e3:.3f} ms",
            "throughput": f"{entry['primary_value']:.2f} {entry['unit']}",
        }
        for entry in results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
