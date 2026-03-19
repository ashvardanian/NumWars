#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-pairs similarity matrix benchmarks: NumKong vs SciPy.

Computes pairwise distance matrices using packed kernels (angulars_packed / euclideans_packed) for NumKong and cdist for SciPy. Both libraries produce (batch_size, batch_size) output matrices.

Can be run with uv:
    uv run --with numkong,numpy,scipy,tabulate similarities/bench.py

Or with traditional pip:
    pip install -e ".[similarities]"
    python similarities/bench.py

Environment variables:
    NUMWARS_FILTER - Regex filter for benchmark names
    NUMWARS_DIMS - Vector dimension and row count (default: 2048)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.spatial.distance as spd
import tabulate

try:
    import numkong as nk
except ImportError:
    print("Error: numkong not found. Install with: pip install numkong")
    sys.exit(1)

try:
    from utils import (
        add_common_args,
        calculate_gso_per_sec,
        format_duration,
        get_vector_dims,
        measure_average_duration,
        normalize_dtype_name,
        numkong_dtype_name,
        parse_numpy_dtype,
        print_results_table,
        should_run_benchmark,
    )
except ImportError:
    print("Error: Could not import utils.py. Make sure it's in the parent directory.")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    library: str
    metric: str
    input_dtype: str
    output_dtype: str
    display_signature: str
    ndim: int
    batch_size: int
    duration_secs: float
    gso_per_sec: float


def display_signature_name(input_dtype: str, output_dtype: str) -> str:
    return f"{input_dtype} \u2192 {output_dtype}"



def build_matrices(batch_size: int, ndim: int, dtype_str: str = "f32", seed: int = 42):
    """Build two random matrices of shape (batch_size, ndim) in the given dtype."""
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)
    if dtype_str.startswith("f") or dtype_str in ("bf16",):
        A = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(np.float32).astype(dtype)
        B = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(np.float32).astype(dtype)
    elif dtype_str.startswith("i"):
        A = rng.integers(-100, 100, size=(batch_size, ndim), dtype=dtype)
        B = rng.integers(-100, 100, size=(batch_size, ndim), dtype=dtype)
    elif dtype_str.startswith("u"):
        A = rng.integers(0, 200, size=(batch_size, ndim), dtype=dtype)
        B = rng.integers(0, 200, size=(batch_size, ndim), dtype=dtype)
    else:
        A = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(dtype)
        B = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(dtype)
    return A, B


# Map from our metric names to scipy metric names
SCIPY_METRIC_MAP = {
    "angular": "cosine",
    "euclidean": "euclidean",
}


def benchmark_scipy(
    metric: str,
    batch_size: int,
    ndim: int,
    seed: int,
    warmup: float,
    profile: float,
) -> BenchmarkResult:
    A, B = build_matrices(batch_size, ndim, "f32", seed)
    out = np.empty((batch_size, batch_size), dtype=np.float64)
    scipy_metric = SCIPY_METRIC_MAP[metric]

    duration = measure_average_duration(
        lambda: spd.cdist(A, B, scipy_metric, out=out),
        warmup,
        profile,
    )
    num_operations = batch_size * batch_size * ndim
    return BenchmarkResult(
        library="SciPy",
        metric=metric,
        input_dtype="f32",
        output_dtype="f64",
        display_signature=display_signature_name("f32", "f64"),
        ndim=ndim,
        batch_size=batch_size,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


PACK_DTYPE_MAP = {"i8": "int8", "u8": "uint8"}


def benchmark_numkong(
    metric: str,
    batch_size: int,
    ndim: int,
    dtype_str: str,
    seed: int,
    warmup: float,
    profile: float,
) -> BenchmarkResult:
    A, B = build_matrices(batch_size, ndim, dtype_str, seed)

    # Pack B once (maps short dtype names to the ones dots_pack expects)
    pack_dtype = PACK_DTYPE_MAP.get(dtype_str, dtype_str)
    B_packed = nk.dots_pack(B, dtype=pack_dtype)

    # Pick the correct packed function
    func = nk.angulars_packed if metric == "angular" else nk.euclideans_packed

    # Discover output dtype from a warm-up call
    sample = func(A, B_packed)
    out_dtype = normalize_dtype_name(getattr(sample, "dtype", "f32"))
    out_array = np.empty((batch_size, batch_size), dtype=parse_numpy_dtype(out_dtype))
    out_tensor = nk.Tensor(out_array, dtype=numkong_dtype_name(out_dtype))

    duration = measure_average_duration(
        lambda: func(A, B_packed, out=out_tensor),
        warmup,
        profile,
    )
    num_operations = batch_size * batch_size * ndim
    return BenchmarkResult(
        library="NumKong",
        metric=metric,
        input_dtype=dtype_str,
        output_dtype=out_dtype,
        display_signature=display_signature_name(dtype_str, out_dtype),
        ndim=ndim,
        batch_size=batch_size,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "similarities",
        "workload": result.metric,
        "benchmark_id": f"similarities/{result.metric}/{result.library.lower()}/{result.input_dtype}",
        "library": result.library,
        "metric": result.metric,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_dtype": result.display_signature,
        "display_signature": result.display_signature,
        "ndim": result.ndim,
        "batch_size": result.batch_size,
        "primary_value": result.gso_per_sec,
        "unit": "GSO/s",
        "duration_secs": result.duration_secs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all-pairs similarity matrices (packed kernels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        "--ndim",
        type=int,
        default=None,
        help="Vector dimensions (default: NUMWARS_DIMS or 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of vectors per matrix (default: NUMWARS_DIMS or 2048)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Choose human-readable table output or machine-readable JSON (default: table).",
    )
    args = parser.parse_args()

    ndim = args.ndim if args.ndim is not None else get_vector_dims()
    batch_size = args.batch_size if args.batch_size is not None else get_batch_size()

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{args.filter}': {e}")

    metadata = {
        "ndim": ndim,
        "batch_size": batch_size,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
    }
    try:
        import scipy as sp
        metadata["scipy_version"] = sp.__version__
    except Exception:
        pass

    candidates = [
        ("numkong", "angular", "f32"),
        ("numkong", "angular", "bf16"),
        ("numkong", "angular", "i8"),
        ("numkong", "angular", "u8"),
        ("scipy", "angular", "f32"),
        ("numkong", "euclidean", "f32"),
        ("numkong", "euclidean", "bf16"),
        ("numkong", "euclidean", "i8"),
        ("numkong", "euclidean", "u8"),
        ("scipy", "euclidean", "f32"),
    ]

    all_results: List[BenchmarkResult] = []
    for library_slug, metric, dtype in candidates:
        benchmark_name = f"similarities/{metric}/{library_slug}/{dtype}"
        if not should_run_benchmark(benchmark_name, filter_pattern):
            continue
        if args.output_format == "table":
            print(f"Benchmarking {benchmark_name}")
        try:
            if library_slug == "numkong":
                all_results.append(
                    benchmark_numkong(metric, batch_size, ndim, dtype, args.seed, args.warmup, args.time_limit)
                )
            else:
                all_results.append(
                    benchmark_scipy(metric, batch_size, ndim, args.seed, args.warmup, args.time_limit)
                )
        except Exception as e:
            if args.output_format == "table":
                print(f"  Error: {e}")

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "suite": "similarities",
                    "settings": metadata,
                    "results": [result_to_entry(r) for r in all_results],
                },
                indent=2,
            )
        )
        return

    if not all_results:
        print("No benchmarks were run.")
        return

    print()
    print(f"All-pairs similarity: A({batch_size}x{ndim}) vs B({batch_size}x{ndim}) -> ({batch_size}x{batch_size})")
    print()
    table_rows = [
        {
            "Library": result.library,
            "Metric": result.metric,
            "Precision": result.display_signature,
            "GSO/s": f"{result.gso_per_sec:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
