#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaxSim (ColBERT-style late-interaction) benchmarks: NumKong vs NumPy.

Computes score = Sigma_i max_j dot(q_i, d_j) using NumKong's maxsim API
and a NumPy matmul + row-max + sum baseline.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,ml_dtypes maxsim/bench.py

Or with traditional pip:
    pip install -e ".[maxsim]"
    python maxsim/bench.py

Environment variables:
    NUMWARS_FILTER - Regex filter for benchmark names
    NUMWARS_DIMS_WIDTH - Document count (default: 2048)
    NUMWARS_DIMS_HEIGHT - Query count (default: 2048)
    NUMWARS_DIMS_DEPTH - Shared dimension (default: 2048)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utils first — it sets single-threaded env vars (OMP_NUM_THREADS, etc.)
# that must be in place before numpy/OpenBLAS initializes.
try:
    from utils import (
        add_common_args,
        calculate_gso_per_sec,
        format_duration,
        get_matrix_dims_depth,
        get_matrix_dims_height,
        get_matrix_dims_width,
        measure_average_duration,
        numkong_dtype_name,
        parse_numpy_dtype,
        print_results_table,
        should_run_benchmark,
    )
except ImportError:
    print("Error: Could not import utils.py. Make sure it's in the parent directory.")
    sys.exit(1)

import numpy as np
import tabulate

try:
    import numkong as nk
except ImportError:
    print("Error: numkong not found. Install with: pip install numkong")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    library: str
    input_dtype: str
    output_dtype: str
    display_signature: str
    height: int
    width: int
    depth: int
    duration_secs: float
    gso_per_sec: float


ACCUMULATOR_DTYPE = {
    "f16": "f32",
    "bf16": "f32",
    "f32": "f64",
}


def display_signature_name(input_dtype: str, output_dtype: str) -> str:
    return f"{input_dtype} \u2192 {output_dtype}"


def build_matrix(shape: tuple[int, int], dtype_str: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)
    data = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return data.astype(dtype)


def benchmark_numkong(
    height: int, width: int, depth: int, dtype_str: str, warmup: float, profile: float
) -> BenchmarkResult:
    queries = build_matrix((height, depth), dtype_str, 42)
    documents = build_matrix((width, depth), dtype_str, 43)
    nk_dtype = numkong_dtype_name(dtype_str)
    packed_queries = nk.maxsim_pack(queries, dtype=nk_dtype)
    packed_documents = nk.maxsim_pack(documents, dtype=nk_dtype)
    output_dtype = ACCUMULATOR_DTYPE.get(dtype_str, dtype_str)

    duration = measure_average_duration(
        lambda: nk.maxsim_packed(packed_queries, packed_documents),
        warmup,
        profile,
    )
    num_operations = 2 * height * width * depth
    return BenchmarkResult(
        library="NumKong",
        input_dtype=dtype_str,
        output_dtype=output_dtype,
        display_signature=display_signature_name(dtype_str, output_dtype),
        height=height,
        width=width,
        depth=depth,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


def benchmark_numpy(
    height: int, width: int, depth: int, warmup: float, profile: float
) -> BenchmarkResult:
    queries = build_matrix((height, depth), "f32", 42)
    documents = build_matrix((width, depth), "f32", 43)
    out = np.empty((height, width), dtype=np.float32)

    def _numpy_maxsim():
        np.matmul(queries, documents.T, out=out)
        return out.max(axis=1).sum()

    duration = measure_average_duration(_numpy_maxsim, warmup, profile)
    num_operations = 2 * height * width * depth
    return BenchmarkResult(
        library="NumPy",
        input_dtype="f32",
        output_dtype="f32",
        display_signature=display_signature_name("f32", "f32"),
        height=height,
        width=width,
        depth=depth,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "maxsim",
        "workload": "maxsim",
        "benchmark_id": f"maxsim/{result.library.lower()}/{result.input_dtype}/{result.height}x{result.width}x{result.depth}",
        "library": result.library,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_dtype": result.display_signature,
        "display_signature": result.display_signature,
        "accumulator_dtype": result.output_dtype,
        "height": result.height,
        "width": result.width,
        "depth": result.depth,
        "primary_value": result.gso_per_sec,
        "unit": "GSO/s",
        "duration_secs": result.duration_secs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MaxSim (ColBERT-style late-interaction) scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Choose human-readable table output or machine-readable JSON (default: table).",
    )
    args = parser.parse_args()

    height = get_matrix_dims_height()
    width = get_matrix_dims_width()
    depth = get_matrix_dims_depth()

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{args.filter}': {e}")

    metadata = {
        "height": height,
        "width": width,
        "depth": depth,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
    }

    candidates = [
        ("numkong", "f16"),
        ("numkong", "bf16"),
        ("numkong", "f32"),
        ("numpy", "f32"),
    ]

    all_results: List[BenchmarkResult] = []
    for library_slug, dtype in candidates:
        benchmark_name = f"maxsim/{library_slug}/{dtype}/{height}x{width}x{depth}"
        if not should_run_benchmark(benchmark_name, filter_pattern):
            continue
        if args.output_format == "table":
            print(f"Benchmarking {benchmark_name}")
        if library_slug == "numkong":
            all_results.append(
                benchmark_numkong(
                    height, width, depth, dtype, args.warmup, args.time_limit
                )
            )
        else:
            all_results.append(
                benchmark_numpy(height, width, depth, args.warmup, args.time_limit)
            )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "suite": "maxsim",
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

    print(f"MaxSim scoring: Q({height}x{depth}) vs D({width}x{depth})")
    print()
    table_rows = [
        {
            "Library": result.library,
            "Precision": result.display_signature,
            "GSO/s": f"{result.gso_per_sec:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
