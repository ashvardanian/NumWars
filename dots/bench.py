#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix multiplication (GEMM) benchmarks: NumKong vs NumPy, PyTorch, JAX, TensorFlow.

Computes C = A @ B.T (NT layout, NVIDIA convention).

Can be run with uv:
    uv run --with numkong,numpy,tabulate dots/bench.py

Or with traditional pip:
    pip install -e ".[dots]"
    python dots/bench.py

Environment variables:
    NUMWARS_FILTER - Regex filter for benchmark names
    NUMWARS_DIMS_WIDTH - Matrix C width (n, default: 1024)
    NUMWARS_DIMS_HEIGHT - Matrix C height (m, default: 1024)
    NUMWARS_DIMS_DEPTH - Shared dimension (k, default: 1024)
    NUMWARS_THREADS - Thread count (default: num_cpus)
    NUMWARS_WARMUP_SECONDS - Warmup time (default: 3.0)
    NUMWARS_PROFILE_SECONDS - Profiling time (default: 10.0)
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Force single-threaded execution for fair comparisons
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import numkong as nk
    HAS_NUMKONG = True
except ImportError:
    HAS_NUMKONG = False
    print("Warning: numkong not found. Install with: pip install numkong")

try:
    import torch
    HAS_TORCH = True  # Enable PyTorch if available
except ImportError:
    HAS_TORCH = False

try:
    from utils import (
        add_common_args,
        calculate_gflops,
        calculate_tops,
        format_duration,
        get_env_list,
        get_env_parsed,
        measure_latency,
        parse_numpy_dtype,
        print_results_table,
        should_run_benchmark,
    )
except ImportError:
    print("Error: Could not import utils.py. Make sure it's in the parent directory.")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    library: str
    dtype: str
    m: int
    n: int
    k: int
    duration_secs: float
    gflops: float
    tops: float
    throughput_gbs: float


# Import dimension helpers from utils
from utils import get_matrix_width, get_matrix_height, get_matrix_depth


def benchmark_numpy_matmul(
    a: np.ndarray, b: np.ndarray, warmup: int = 3, iterations: int = 10
) -> float:
    """Benchmark NumPy matrix multiplication A @ B.T"""
    # Warmup
    for _ in range(warmup):
        _ = a @ b.T

    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        result = a @ b.T
    end = time.perf_counter()

    return (end - start) / iterations


def benchmark_numkong_matmul(
    a: np.ndarray, b: np.ndarray, warmup: int = 3, iterations: int = 10
) -> float:
    """Benchmark NumKong matrix multiplication (if available)"""
    if not HAS_NUMKONG:
        return float("inf")

    # Warmup
    for _ in range(warmup):
        # NumKong doesn't have direct matmul yet, use dot products in a loop
        # This is a placeholder - actual implementation will depend on NumKong API
        _ = a @ b.T

    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        result = a @ b.T
    end = time.perf_counter()

    return (end - start) / iterations


def benchmark_torch_matmul(
    a: np.ndarray, b: np.ndarray, warmup: int = 3, iterations: int = 10
) -> float:
    """Benchmark PyTorch matrix multiplication A @ B.T"""
    if not HAS_TORCH:
        return float("inf")

    a_torch = torch.from_numpy(a)
    b_torch = torch.from_numpy(b)

    # Warmup
    for _ in range(warmup):
        _ = a_torch @ b_torch.T

    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        result = a_torch @ b_torch.T
    end = time.perf_counter()

    return (end - start) / iterations


def run_benchmark(
    m: int,
    n: int,
    k: int,
    dtype_str: str,
    warmup_time: float = 1.0,
    measurement_time: float = 5.0,
) -> List[BenchmarkResult]:
    """Run benchmarks for a specific matrix size and data type."""
    results = []

    try:
        dtype = parse_numpy_dtype(dtype_str)
    except ValueError as e:
        print(f"Skipping {dtype_str}: {e}")
        return results

    # Generate random matrices: A (m×k), B (n×k) for A @ B.T → (m×n)
    print(f"  Generating matrices: A({m}×{k}) @ B.T({n}×{k}) → ({m}×{n}), dtype={dtype_str}")
    rng = np.random.default_rng(42)

    if dtype_str.startswith("f") or dtype_str in ("bf16", "e4m3", "e5m2"):
        a = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32).astype(dtype)
        b = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32).astype(dtype)
    else:
        # Integer types
        a = rng.integers(-100, 100, size=(m, k)).astype(dtype)
        b = rng.integers(-100, 100, size=(n, k)).astype(dtype)

    # Calculate metrics
    num_operations = 2 * m * n * k  # 2 operations per multiply-add
    bytes_processed = (m * k + n * k + m * n) * a.itemsize

    # Benchmark NumPy
    print("    NumPy...", end=" ", flush=True)
    duration_numpy = benchmark_numpy_matmul(a, b, warmup=3, iterations=max(1, int(measurement_time)))
    gflops_numpy = calculate_gflops(num_operations, duration_numpy)
    tops_numpy = calculate_tops(num_operations, duration_numpy)
    throughput_numpy = bytes_processed / duration_numpy / 1e9

    results.append(
        BenchmarkResult(
            library="NumPy",
            dtype=dtype_str,
            m=m,
            n=n,
            k=k,
            duration_secs=duration_numpy,
            gflops=gflops_numpy,
            tops=tops_numpy,
            throughput_gbs=throughput_numpy,
        )
    )
    print(f"{gflops_numpy:.2f} GFLOPS")

    # Benchmark NumKong (if available)
    if HAS_NUMKONG:
        print("    NumKong...", end=" ", flush=True)
        duration_numkong = benchmark_numkong_matmul(a, b, warmup=3, iterations=max(1, int(measurement_time)))
        gflops_numkong = calculate_gflops(num_operations, duration_numkong)
        tops_numkong = calculate_tops(num_operations, duration_numkong)
        throughput_numkong = bytes_processed / duration_numkong / 1e9

        results.append(
            BenchmarkResult(
                library="NumKong",
                dtype=dtype_str,
                m=m,
                n=n,
                k=k,
                duration_secs=duration_numkong,
                gflops=gflops_numkong,
                tops=tops_numkong,
                throughput_gbs=throughput_numkong,
            )
        )
        print(f"{gflops_numkong:.2f} GFLOPS")

    # Benchmark PyTorch (if available and enabled)
    if HAS_TORCH:
        print("    PyTorch...", end=" ", flush=True)
        duration_torch = benchmark_torch_matmul(a, b, warmup=3, iterations=max(1, int(measurement_time)))
        gflops_torch = calculate_gflops(num_operations, duration_torch)
        tops_torch = calculate_tops(num_operations, duration_torch)
        throughput_torch = bytes_processed / duration_torch / 1e9

        results.append(
            BenchmarkResult(
                library="PyTorch",
                dtype=dtype_str,
                m=m,
                n=n,
                k=k,
                duration_secs=duration_torch,
                gflops=gflops_torch,
                tops=tops_torch,
                throughput_gbs=throughput_torch,
            )
        )
        print(f"{gflops_torch:.2f} GFLOPS")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark matrix multiplication (GEMM) operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Get matrix dimensions from environment variables
    m = get_matrix_dims_height()
    n = get_matrix_dims_width()
    k = get_matrix_dims_depth()

    # Always benchmark all dtypes - filtering via NUMWARS_FILTER
    dtypes = ["f32", "f64"]

    # Compile filter pattern if provided
    import re
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{args.filter}': {e}")
            filter_pattern = None

    print(f"Matrix Multiplication (GEMM) Benchmarks")
    print(f"Matrix dimensions: {m}×{n}×{k} (C = A @ B.T where A is {m}×{k}, B is {n}×{k})")
    print(f"Data types: {dtypes}")
    print()

    all_results = []

    for dtype in dtypes:
        # Construct hierarchical benchmark name: dots/{library}/{dtype}/{m}x{n}x{k}
        benchmark_name = f"dots/numpy/{dtype}/{m}x{n}x{k}"

        if not should_run_benchmark(benchmark_name, filter_pattern):
            continue

        print(f"Benchmarking {m}×{k} @ {n}×{k}.T = {m}×{n} ({dtype}):")
        results = run_benchmark(
            m, n, k, dtype, warmup_time=args.warmup, measurement_time=args.time_limit
        )
        all_results.extend(results)
        print()

    # Print summary table
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80 + "\n")

        table_data = []
        for r in all_results:
            table_data.append(
                {
                    "Library": r.library,
                    "DType": r.dtype,
                    "Size": f"{r.m}×{r.n}",
                    "GFLOPS": f"{r.gflops:.2f}",
                    "GB/s": f"{r.throughput_gbs:.2f}",
                    "Time": format_duration(r.duration_secs),
                }
            )

        print_results_table(table_data)
    else:
        print("No benchmarks were run.")


if __name__ == "__main__":
    main()
