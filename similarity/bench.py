#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairwise similarity benchmarks: NumKong vs NumPy vs SciPy.

Measures dot, angular, euclidean, and sqeuclidean distances between vector pairs.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,ml_dtypes similarity/bench.py

Or with traditional pip:
    pip install -e ".[similarity]"
    python similarity/bench.py

Environment variables:
    NUMWARS_FILTER - Regex filter for benchmark names
    NUMWARS_DIMS - Vector dimension (default: 2048)
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
        parse_numpy_dtype,
        print_results_table,
        should_run_benchmark,
    )
except ImportError:
    print("Error: Could not import utils.py. Make sure it's in the parent directory.")
    sys.exit(1)

# Suppress floating-point warnings (overflow in half-precision, etc.)
np.seterr(all="ignore")


@dataclass
class BenchmarkResult:
    library: str
    metric: str
    input_dtype: str
    output_dtype: str
    display_signature: str
    ndim: int
    count: int
    duration_secs: float
    gso_per_sec: float


NUM_MATRIX_PAIRS_ = 16
NUM_REPS_PER_PAIR_ = 3


def display_signature(input_dtype: str, output_dtype: str) -> str:
    return f"{input_dtype} \u2192 {output_dtype}"



def random_matrix(count: int, ndim: int, dtype_str: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)
    if dtype_str.startswith("f") or dtype_str in ("bf16",):
        data = rng.uniform(-1.0, 1.0, size=(count, ndim)).astype(np.float32)
        return data.astype(dtype)
    elif dtype_str.startswith("i"):
        return rng.integers(-100, 100, size=(count, ndim), dtype=dtype)
    elif dtype_str.startswith("u"):
        return rng.integers(0, 200, size=(count, ndim), dtype=dtype)
    else:
        return rng.uniform(-1.0, 1.0, size=(count, ndim)).astype(dtype)


_NK_METRIC_FUNC = {
    "dot": nk.dot,
    "angular": nk.angular,
    "euclidean": nk.euclidean,
    "sqeuclidean": nk.sqeuclidean,
}


def _nk_pairwise(metric: str, a, b, dtype_str: str):
    """Call the appropriate numkong pairwise function."""
    func = _NK_METRIC_FUNC[metric]
    if dtype_str in ("bf16",):
        return func(a, b, nk.bfloat16)
    return func(a, b)


def benchmark_numkong_pairwise(
    metric: str, ndim: int, count: int, dtype_str: str, warmup: float, profile: float
) -> BenchmarkResult:
    matrices = [
        random_matrix(count, ndim, dtype_str, seed=i) for i in range(NUM_MATRIX_PAIRS_)
    ]
    if count == 1:
        matrices = [m.flatten() for m in matrices]

    # Warm up to discover output dtype
    sample = _nk_pairwise(metric, matrices[0], matrices[1], dtype_str)
    out_dtype = normalize_dtype_name(getattr(sample, "dtype", "f32"))

    def run():
        for i in range(1, NUM_MATRIX_PAIRS_):
            for _ in range(NUM_REPS_PER_PAIR_):
                _nk_pairwise(metric, matrices[i - 1], matrices[i], dtype_str)

    duration = measure_average_duration(run, warmup, profile)
    total_calls = (NUM_MATRIX_PAIRS_ - 1) * NUM_REPS_PER_PAIR_
    distance_calculations = count * total_calls
    per_call = duration / total_calls
    gso = (distance_calculations * ndim) / duration / 1e9

    return BenchmarkResult(
        library="NumKong",
        metric=metric,
        input_dtype=dtype_str,
        output_dtype=out_dtype,
        display_signature=display_signature(dtype_str, out_dtype),
        ndim=ndim,
        count=count,
        duration_secs=per_call,
        gso_per_sec=gso,
    )


def benchmark_scipy_pairwise(
    metric: str, ndim: int, count: int, dtype_str: str, warmup: float, profile: float
) -> BenchmarkResult:
    import scipy.spatial.distance as spd

    matrices = [
        random_matrix(count, ndim, dtype_str, seed=i) for i in range(NUM_MATRIX_PAIRS_)
    ]
    if count == 1:
        matrices = [m.flatten() for m in matrices]

    scipy_metric_map = {
        "angular": "cosine",
        "euclidean": "euclidean",
        "sqeuclidean": "sqeuclidean",
    }
    sp_metric = scipy_metric_map[metric]

    def one_to_one(a, b):
        return getattr(spd, sp_metric)(a, b)

    if count == 1:
        call_fn = one_to_one
    else:

        def call_fn(a, b):
            for i in range(a.shape[0]):
                getattr(spd, sp_metric)(a[i], b[i])

    def run():
        for i in range(1, NUM_MATRIX_PAIRS_):
            for _ in range(NUM_REPS_PER_PAIR_):
                call_fn(matrices[i - 1], matrices[i])

    duration = measure_average_duration(run, warmup, profile)
    total_calls = (NUM_MATRIX_PAIRS_ - 1) * NUM_REPS_PER_PAIR_
    distance_calculations = count * total_calls
    per_call = duration / total_calls
    gso = (distance_calculations * ndim) / duration / 1e9

    return BenchmarkResult(
        library="SciPy",
        metric=metric,
        input_dtype=dtype_str,
        output_dtype="f64",
        display_signature=display_signature(dtype_str, "f64"),
        ndim=ndim,
        count=count,
        duration_secs=per_call,
        gso_per_sec=gso,
    )


def benchmark_scipy_dot_pairwise(
    ndim: int, count: int, dtype_str: str, warmup: float, profile: float
) -> BenchmarkResult:
    import scipy.linalg.blas as spb

    matrices = [
        random_matrix(count, ndim, dtype_str, seed=i) for i in range(NUM_MATRIX_PAIRS_)
    ]
    if count == 1:
        matrices = [m.flatten() for m in matrices]

    if count == 1:
        call_fn = spb.sdot
    else:

        def call_fn(a, b):
            for i in range(a.shape[0]):
                spb.sdot(a[i], b[i])

    def run():
        for i in range(1, NUM_MATRIX_PAIRS_):
            for _ in range(NUM_REPS_PER_PAIR_):
                call_fn(matrices[i - 1], matrices[i])

    duration = measure_average_duration(run, warmup, profile)
    total_calls = (NUM_MATRIX_PAIRS_ - 1) * NUM_REPS_PER_PAIR_
    distance_calculations = count * total_calls
    per_call = duration / total_calls
    gso = (distance_calculations * ndim) / duration / 1e9

    return BenchmarkResult(
        library="SciPy",
        metric="dot",
        input_dtype=dtype_str,
        output_dtype="f32",
        display_signature=display_signature(dtype_str, "f32"),
        ndim=ndim,
        count=count,
        duration_secs=per_call,
        gso_per_sec=gso,
    )


def benchmark_numpy_dot_pairwise(
    ndim: int, count: int, dtype_str: str, warmup: float, profile: float
) -> BenchmarkResult:
    matrices = [
        random_matrix(count, ndim, dtype_str, seed=i) for i in range(NUM_MATRIX_PAIRS_)
    ]
    if count == 1:
        matrices = [m.flatten() for m in matrices]

    if count == 1:
        call_fn = np.dot
    else:

        def call_fn(a, b):
            np.sum(a * b, axis=1)

    def run():
        for i in range(1, NUM_MATRIX_PAIRS_):
            for _ in range(NUM_REPS_PER_PAIR_):
                call_fn(matrices[i - 1], matrices[i])

    duration = measure_average_duration(run, warmup, profile)
    total_calls = (NUM_MATRIX_PAIRS_ - 1) * NUM_REPS_PER_PAIR_
    distance_calculations = count * total_calls
    per_call = duration / total_calls
    gso = (distance_calculations * ndim) / duration / 1e9

    return BenchmarkResult(
        library="NumPy",
        metric="dot",
        input_dtype=dtype_str,
        output_dtype=dtype_str,
        display_signature=display_signature(dtype_str, dtype_str),
        ndim=ndim,
        count=count,
        duration_secs=per_call,
        gso_per_sec=gso,
    )


CANDIDATES = [
    # (library, metric, dtype)
    ("numkong", "dot", "f32"),
    ("numkong", "dot", "f64"),
    ("numkong", "dot", "i8"),
    ("numkong", "dot", "u8"),
    ("numkong", "dot", "bf16"),
    ("numkong", "angular", "f32"),
    ("numkong", "angular", "f64"),
    ("numkong", "angular", "i8"),
    ("numkong", "angular", "u8"),
    ("numkong", "angular", "bf16"),
    ("numkong", "euclidean", "f32"),
    ("numkong", "euclidean", "f64"),
    ("numkong", "euclidean", "i8"),
    ("numkong", "euclidean", "u8"),
    ("numkong", "euclidean", "bf16"),
    ("numkong", "sqeuclidean", "f32"),
    ("numkong", "sqeuclidean", "f64"),
    ("numkong", "sqeuclidean", "i8"),
    ("numkong", "sqeuclidean", "u8"),
    ("numkong", "sqeuclidean", "bf16"),
    ("scipy", "dot", "f32"),
    ("scipy", "angular", "f32"),
    ("scipy", "euclidean", "f32"),
    ("scipy", "sqeuclidean", "f32"),
    ("numpy", "dot", "f32"),
    ("numpy", "dot", "f64"),
]


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "similarity",
        "workload": result.metric,
        "benchmark_id": f"similarity/{result.metric}/{result.input_dtype}/{result.library.lower()}",
        "library": result.library,
        "metric": result.metric,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_signature": result.display_signature,
        "ndim": result.ndim,
        "count": result.count,
        "primary_value": result.gso_per_sec,
        "unit": "GSO/s",
        "duration_secs": result.duration_secs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pairwise similarity: NumKong vs NumPy vs SciPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        "--ndim",
        type=int,
        default=get_vector_dims(),
        help="Number of vector dimensions (default: from NUMWARS_DIMS or 2048)",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of vectors per batch (default: 1)",
    )
    parser.add_argument(
        "--scipy",
        action="store_true",
        help="Include SciPy benchmarks (must be installed)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    ndim = args.ndim
    count = args.count

    assert ndim > 0, "Vector dimensions must be > 0"
    assert count > 0, "Count must be > 0"

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{args.filter}': {e}")

    metadata = {
        "ndim": ndim,
        "count": count,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
    }

    all_results: List[BenchmarkResult] = []
    for library, metric, dtype_str in CANDIDATES:
        if library == "scipy" and not args.scipy:
            continue

        benchmark_name = f"similarity/{metric}/{dtype_str}"
        if not should_run_benchmark(benchmark_name, filter_pattern):
            continue

        if args.output_format == "table":
            print(f"Benchmarking {library}/{metric}/{dtype_str}")

        try:
            if library == "numkong":
                result = benchmark_numkong_pairwise(
                    metric, ndim, count, dtype_str, args.warmup, args.time_limit
                )
            elif library == "scipy" and metric == "dot":
                result = benchmark_scipy_dot_pairwise(
                    ndim, count, dtype_str, args.warmup, args.time_limit
                )
            elif library == "scipy":
                result = benchmark_scipy_pairwise(
                    metric, ndim, count, dtype_str, args.warmup, args.time_limit
                )
            elif library == "numpy":
                result = benchmark_numpy_dot_pairwise(
                    ndim, count, dtype_str, args.warmup, args.time_limit
                )
            else:
                continue

            all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "suite": "similarity",
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
    print(f"Similarity benchmarks: ndim={ndim}, count={count}")
    print()

    table_rows = [
        {
            "Library": r.library,
            "Metric": r.metric,
            "Precision": r.display_signature,
            "GSO/s": f"{r.gso_per_sec:.2f}",
            "Time": format_duration(r.duration_secs),
        }
        for r in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
