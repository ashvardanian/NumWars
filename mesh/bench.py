#!/usr/bin/env python3
"""
Mesh alignment benchmarks: NumKong vs Python alternatives.

Benchmarks RMSD and Kabsch alignment over 3D point clouds:
- NumKong: nk.rmsd / nk.kabsch
- NumPy: manual RMSD via array ops
- BioPython: SVDSuperimposer (Kabsch)

Can be run with uv:
    uv run --with numkong,numpy,biopython,tabulate mesh/bench.py

Or with traditional pip:
    pip install -e ".[mesh]"
    python mesh/bench.py
"""

import json
import sys
from dataclasses import dataclass

import numpy as np

from numwars import (
    Settings,
    calculate_mps,
    measure_average_duration,
    normalize_dtype_name,
    print_results_table,
    print_settings,
    read_settings,
)

try:
    import numkong as nk
except ImportError:
    print("Error: numkong not found. Install with: pip install numkong")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    workload: str
    library: str
    input_dtype: str
    output_dtype: str
    count: int
    duration_secs: float
    primary_value: float


def build_point_clouds(count: int, seed: int, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Generate two random (N, 3) point clouds of the given dtype."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((count, 3)).astype(dtype)
    target = rng.standard_normal((count, 3)).astype(dtype)
    return source, target


def benchmark_numkong_rmsd(
    source: np.ndarray,
    target: np.ndarray,
    settings: Settings,
) -> BenchmarkResult:
    count = source.shape[0]
    duration = measure_average_duration(
        lambda: nk.rmsd(source, target),
        settings,
    )
    return BenchmarkResult(
        workload="rmsd",
        library="NumKong",
        input_dtype=normalize_dtype_name(source.dtype),
        output_dtype="f64",
        count=count,
        duration_secs=duration,
        primary_value=calculate_mps(count, duration),
    )


def benchmark_numpy_rmsd(
    source: np.ndarray,
    target: np.ndarray,
    settings: Settings,
) -> BenchmarkResult:
    count = source.shape[0]
    duration = measure_average_duration(
        lambda: np.sqrt(np.mean(np.sum((source - target) ** 2, axis=1))),
        settings,
    )
    return BenchmarkResult(
        workload="rmsd",
        library="NumPy",
        input_dtype=normalize_dtype_name(source.dtype),
        output_dtype="f64",
        count=count,
        duration_secs=duration,
        primary_value=calculate_mps(count, duration),
    )


def benchmark_numkong_kabsch(
    source: np.ndarray,
    target: np.ndarray,
    settings: Settings,
) -> BenchmarkResult:
    count = source.shape[0]
    duration = measure_average_duration(
        lambda: nk.kabsch(source, target),
        settings,
    )
    return BenchmarkResult(
        workload="kabsch",
        library="NumKong",
        input_dtype=normalize_dtype_name(source.dtype),
        output_dtype="f64",
        count=count,
        duration_secs=duration,
        primary_value=calculate_mps(count, duration),
    )


def benchmark_biopython_kabsch(
    source: np.ndarray,
    target: np.ndarray,
    settings: Settings,
) -> BenchmarkResult:
    try:
        from Bio.SVDSuperimposer import SVDSuperimposer
    except ImportError:
        print("Warning: biopython not found, skipping biopython kabsch benchmark.")
        return None

    sup = SVDSuperimposer()
    count = source.shape[0]

    def run_once():
        sup.set(target, source)
        sup.run()

    duration = measure_average_duration(run_once, settings)
    return BenchmarkResult(
        workload="kabsch",
        library="BioPython",
        input_dtype=normalize_dtype_name(source.dtype),
        output_dtype="f64",
        count=count,
        duration_secs=duration,
        primary_value=calculate_mps(count, duration),
    )


def benchmark_numkong_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    settings: Settings,
) -> BenchmarkResult:
    count = source.shape[0]
    duration = measure_average_duration(
        lambda: nk.umeyama(source, target),
        settings,
    )
    return BenchmarkResult(
        workload="umeyama",
        library="NumKong",
        input_dtype=normalize_dtype_name(source.dtype),
        output_dtype="f64",
        count=count,
        duration_secs=duration,
        primary_value=calculate_mps(count, duration),
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "mesh",
        "workload": result.workload,
        "benchmark_id": f"mesh/{result.workload}/{result.library.lower()}/{result.input_dtype}",
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
    settings = read_settings()
    if settings.output == "table":
        print_settings(settings)
    count = settings.batch_size

    source, target = build_point_clouds(count, settings.seed)
    source64, target64 = build_point_clouds(count, settings.seed, dtype=np.float64)
    benchmarks = [
        (
            "mesh/rmsd/numkong/f32",
            lambda: benchmark_numkong_rmsd(source, target, settings),
        ),
        (
            "mesh/rmsd/numkong/f64",
            lambda: benchmark_numkong_rmsd(source64, target64, settings),
        ),
        (
            "mesh/rmsd/numpy/f32",
            lambda: benchmark_numpy_rmsd(source, target, settings),
        ),
        (
            "mesh/rmsd/numpy/f64",
            lambda: benchmark_numpy_rmsd(source64, target64, settings),
        ),
        (
            "mesh/kabsch/numkong/f32",
            lambda: benchmark_numkong_kabsch(source, target, settings),
        ),
        (
            "mesh/kabsch/numkong/f64",
            lambda: benchmark_numkong_kabsch(source64, target64, settings),
        ),
        (
            "mesh/kabsch/biopython/f32",
            lambda: benchmark_biopython_kabsch(source, target, settings),
        ),
        (
            "mesh/kabsch/biopython/f64",
            lambda: benchmark_biopython_kabsch(source64, target64, settings),
        ),
        (
            "mesh/umeyama/numkong/f32",
            lambda: benchmark_numkong_umeyama(source, target, settings),
        ),
        (
            "mesh/umeyama/numkong/f64",
            lambda: benchmark_numkong_umeyama(source64, target64, settings),
        ),
    ]

    results = []
    for benchmark_id, benchmark_fn in benchmarks:
        if not settings.selects(benchmark_id):
            continue
        result = benchmark_fn()
        if result is not None:
            results.append(result_to_entry(result))

    if settings.output == "json":
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
