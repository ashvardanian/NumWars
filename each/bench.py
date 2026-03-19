#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elementwise operation benchmarks: NumKong vs NumPy.

Benchmarks add operations across multiple data types.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,ml_dtypes each/bench.py

Or with traditional pip:
    pip install -e ".[each]"
    python each/bench.py

Environment variables:
    NUMWARS_FILTER - Regex filter for benchmark names
    NUMWARS_DIMS - Tensor size in elements (default: 1000000)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

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
        format_duration,
        get_env,
        get_tensor_dims,
        measure_average_duration,
        normalize_dtype_name,
        parse_numpy_dtype,
        print_results_table,
        should_run_benchmark,
    )
except ImportError:
    print("Error: Could not import utils.py. Make sure it's in the parent directory.")
    sys.exit(1)

# Suppress floating-point warnings during benchmarking
np.seterr(all="ignore")


@dataclass
class BenchmarkResult:
    library: str
    operation: str
    input_dtype: str
    output_dtype: str
    display_signature: str
    elements: int
    duration_secs: float
    throughput_gbs: float


def dtype_itemsize(dtype_name: str) -> int:
    mapping = {
        "f64": 8,
        "f32": 4,
        "f16": 2,
        "bf16": 2,
        "i8": 1,
    }
    if dtype_name not in mapping:
        raise KeyError(f"Unknown dtype itemsize for {dtype_name}")
    return mapping[dtype_name]


def display_signature(input_dtype: str, output_dtype: str) -> str:
    return f"{input_dtype} \u2192 {output_dtype}"


def build_array(elements: int, dtype_str: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype_str == "bf16":
        return rng.uniform(-1.0, 1.0, size=(elements,)).astype(np.float32).astype(parse_numpy_dtype("bf16"))
    np_dtype = parse_numpy_dtype(dtype_str)
    if dtype_str == "i8":
        return rng.integers(-128, 127, size=(elements,), dtype=np_dtype)
    return rng.uniform(-1.0, 1.0, size=(elements,)).astype(np.float32).astype(np_dtype)



def benchmark_numkong_add(
    elements: int, dtype_str: str, warmup: float, profile: float, seed: int = 42,
) -> BenchmarkResult:
    a = build_array(elements, dtype_str, seed)
    b = build_array(elements, dtype_str, seed + 1)

    if dtype_str == "bf16":
        func = lambda: nk.add(a, b, a_dtype="bfloat16", b_dtype="bfloat16", out_dtype="bfloat16")
    else:
        func = lambda: nk.add(a, b)

    duration = measure_average_duration(func, warmup, profile)
    itemsize = dtype_itemsize(dtype_str)
    bytes_processed = 2 * elements * itemsize + elements * itemsize
    return BenchmarkResult(
        library="NumKong",
        operation="add",
        input_dtype=dtype_str,
        output_dtype=dtype_str,
        display_signature=display_signature(dtype_str, dtype_str),
        elements=elements,
        duration_secs=duration,
        throughput_gbs=bytes_processed / duration / 1e9,
    )


def benchmark_numpy_add(
    elements: int, dtype_str: str, warmup: float, profile: float, seed: int = 42,
) -> BenchmarkResult:
    a = build_array(elements, dtype_str, seed)
    b = build_array(elements, dtype_str, seed + 1)
    out = np.empty_like(a)

    func = lambda: np.add(a, b, out=out)

    duration = measure_average_duration(func, warmup, profile)
    itemsize = dtype_itemsize(dtype_str)
    bytes_processed = 2 * elements * itemsize + elements * itemsize
    return BenchmarkResult(
        library="NumPy",
        operation="add",
        input_dtype=dtype_str,
        output_dtype=dtype_str,
        display_signature=display_signature(dtype_str, dtype_str),
        elements=elements,
        duration_secs=duration,
        throughput_gbs=bytes_processed / duration / 1e9,
    )


candidates = [
    # add
    ("numkong", "add", "f32"),
    ("numkong", "add", "f64"),
    ("numkong", "add", "f16"),
    ("numkong", "add", "bf16"),
    ("numkong", "add", "i8"),
    ("numpy", "add", "f32"),
    ("numpy", "add", "f64"),
    ("numpy", "add", "f16"),
    ("numpy", "add", "i8"),
]

dispatch = {
    ("numkong", "add"): benchmark_numkong_add,
    ("numpy", "add"): benchmark_numpy_add,
}


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "each",
        "workload": result.operation,
        "benchmark_id": f"each/{result.operation}/{result.input_dtype}/{result.library}",
        "library": result.library,
        "operation": result.operation,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_signature": result.display_signature,
        "elements": result.elements,
        "primary_value": result.throughput_gbs,
        "unit": "GB/s",
        "throughput_gbs": result.throughput_gbs,
        "duration_secs": result.duration_secs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark elementwise operations: NumKong vs NumPy",
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

    elements = get_tensor_dims()

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{args.filter}': {e}")

    metadata = {
        "elements": elements,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
    }

    all_results: List[BenchmarkResult] = []
    for library_slug, operation, dtype_str in candidates:
        benchmark_name = f"each/{operation}/{dtype_str}"
        if not should_run_benchmark(benchmark_name, filter_pattern):
            continue
        if args.output_format == "table":
            print(f"Benchmarking {benchmark_name} ({library_slug})")

        func = dispatch[(library_slug, operation)]
        result = func(elements, dtype_str, args.warmup, args.time_limit, args.seed)
        all_results.append(result)

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "suite": "each",
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
    print(f"Elementwise operations: {elements:,} elements")
    print()
    table_rows = [
        {
            "Library": result.library,
            "Operation": result.operation,
            "Precision": result.display_signature,
            "GB/s": f"{result.throughput_gbs:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
