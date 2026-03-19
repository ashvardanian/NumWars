#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduction benchmarks: NumKong vs NumPy.

Covers flat-vector sum, min, max, argmin, argmax and row-wise L2 norms.

Can be run with uv:
    uv run --with numkong,numpy,tabulate reduce/bench.py

Or with traditional pip:
    pip install -e ".[reduce]"
    python reduce/bench.py

Environment variables:
    NUMWARS_FILTER - Regex filter for benchmark names
    NUMWARS_DIMS   - Number of elements for flat-vector ops (default: 1_000_000)
    NUMWARS_DIMS - Vector dimension and row count (default: 2048)
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tabulate

try:
    import numkong as nk
except ImportError:
    print("Error: numkong not found. Install with: pip install numkong")
    sys.exit(1)

try:
    from utils import (
        add_common_args,
        format_duration,
        get_batch_size,
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
        "f64": np.dtype(np.float64).itemsize,
        "f32": np.dtype(np.float32).itemsize,
        "f16": np.dtype(np.float16).itemsize,
        "bf16": np.dtype(np.uint16).itemsize,
        "i64": np.dtype(np.int64).itemsize,
        "i32": np.dtype(np.int32).itemsize,
        "i16": np.dtype(np.int16).itemsize,
        "i8": np.dtype(np.int8).itemsize,
        "u64": np.dtype(np.uint64).itemsize,
        "u32": np.dtype(np.uint32).itemsize,
        "u16": np.dtype(np.uint16).itemsize,
        "u8": np.dtype(np.uint8).itemsize,
    }
    if dtype_name not in mapping:
        raise KeyError(f"Unknown dtype itemsize for {dtype_name}")
    return mapping[dtype_name]


def display_signature_name(input_dtype: str, output_dtype: str) -> str:
    return f"{input_dtype} \u2192 {output_dtype}"


def build_vector(elements: int, dtype_str: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)
    if dtype_str.startswith("i") or dtype_str.startswith("u"):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=elements, dtype=dtype)
    data = rng.uniform(-1.0, 1.0, size=elements).astype(np.float32)
    return data.astype(dtype)


def build_matrix(batch_size: int, ndim: int, dtype_str: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)
    data = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(np.float32)
    return data.astype(dtype)


def benchmark_numpy(
    op: str, elements: int, dtype_str: str, warmup: float, profile: float, seed: int
) -> BenchmarkResult:
    if op == "norm":
        batch_size = get_batch_size()
        ndim = elements // batch_size
        if ndim < 1:
            ndim = 1
        actual_elements = batch_size * ndim
        matrix = build_matrix(batch_size, ndim, dtype_str, seed)
        func = lambda: np.linalg.norm(matrix, axis=1)
        bytes_processed = actual_elements * matrix.itemsize
    else:
        actual_elements = elements
        x = build_vector(elements, dtype_str, seed)
        if op == "sum":
            func = lambda: np.sum(x)
        elif op == "min":
            func = lambda: np.min(x)
        elif op == "max":
            func = lambda: np.max(x)
        elif op == "argmin":
            func = lambda: np.argmin(x)
        elif op == "argmax":
            func = lambda: np.argmax(x)
        else:
            raise ValueError(f"Unknown operation: {op}")
        bytes_processed = actual_elements * x.itemsize

    duration = measure_average_duration(func, warmup, profile)
    throughput_gbs = bytes_processed / duration / 1e9 if duration > 0 else 0.0
    output_dtype = dtype_str
    if op in ("argmin", "argmax"):
        output_dtype = "i64"
    elif op == "norm":
        output_dtype = "f64"

    return BenchmarkResult(
        library="NumPy",
        operation=op,
        input_dtype=dtype_str,
        output_dtype=output_dtype,
        display_signature=display_signature_name(dtype_str, output_dtype),
        elements=actual_elements,
        duration_secs=duration,
        throughput_gbs=throughput_gbs,
    )


def benchmark_numkong(
    op: str, elements: int, dtype_str: str, warmup: float, profile: float, seed: int
) -> BenchmarkResult:
    if op == "norm":
        batch_size = get_batch_size()
        ndim = elements // batch_size
        if ndim < 1:
            ndim = 1
        actual_elements = batch_size * ndim
        matrix = build_matrix(batch_size, ndim, dtype_str, seed)
        t = nk.Tensor(matrix)
        func = lambda: nk.norm(t, axis=1)
        bytes_processed = actual_elements * matrix.itemsize
    else:
        actual_elements = elements
        x = build_vector(elements, dtype_str, seed)
        if op == "sum":
            t = nk.Tensor(x)
            func = lambda: nk.sum(t)
        elif op == "min":
            func = lambda: nk.min(x)
        elif op == "max":
            func = lambda: nk.max(x)
        elif op == "argmin":
            func = lambda: nk.argmin(x)
        elif op == "argmax":
            func = lambda: nk.argmax(x)
        else:
            raise ValueError(f"Unknown operation: {op}")
        bytes_processed = actual_elements * x.itemsize

    duration = measure_average_duration(func, warmup, profile)
    throughput_gbs = bytes_processed / duration / 1e9 if duration > 0 else 0.0
    output_dtype = dtype_str
    if op in ("argmin", "argmax"):
        output_dtype = "i64"
    elif op == "norm":
        output_dtype = "f64"

    return BenchmarkResult(
        library="NumKong",
        operation=op,
        input_dtype=dtype_str,
        output_dtype=output_dtype,
        display_signature=display_signature_name(dtype_str, output_dtype),
        elements=actual_elements,
        duration_secs=duration,
        throughput_gbs=throughput_gbs,
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "reduce",
        "workload": result.operation,
        "benchmark_id": f"reduce/{result.operation}/{result.library.lower()}/{result.input_dtype}",
        "library": result.library,
        "operation": result.operation,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_dtype": result.display_signature,
        "display_signature": result.display_signature,
        "elements": result.elements,
        "primary_value": result.throughput_gbs,
        "unit": "GB/s",
        "throughput_gbs": result.throughput_gbs,
        "duration_secs": result.duration_secs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vector reduction operations",
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
        "batch_size": get_batch_size(),
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
    }

    candidates = [
        ("numpy", "sum", "f32"),
        ("numpy", "sum", "f64"),
        ("numpy", "sum", "i8"),
        ("numkong", "sum", "f32"),
        ("numkong", "sum", "f64"),
        ("numkong", "sum", "i8"),
        ("numpy", "norm", "f32"),
        ("numpy", "norm", "f64"),
        ("numkong", "norm", "f32"),
        ("numkong", "norm", "f64"),
        ("numpy", "min", "f32"),
        ("numpy", "min", "f64"),
        ("numpy", "min", "i8"),
        ("numkong", "min", "f32"),
        ("numkong", "min", "f64"),
        ("numkong", "min", "i8"),
        ("numpy", "max", "f32"),
        ("numpy", "max", "f64"),
        ("numpy", "max", "i8"),
        ("numkong", "max", "f32"),
        ("numkong", "max", "f64"),
        ("numkong", "max", "i8"),
        ("numpy", "argmin", "f32"),
        ("numpy", "argmin", "f64"),
        ("numpy", "argmin", "i8"),
        ("numkong", "argmin", "f32"),
        ("numkong", "argmin", "f64"),
        ("numkong", "argmin", "i8"),
        ("numpy", "argmax", "f32"),
        ("numpy", "argmax", "f64"),
        ("numpy", "argmax", "i8"),
        ("numkong", "argmax", "f32"),
        ("numkong", "argmax", "f64"),
        ("numkong", "argmax", "i8"),
    ]

    all_results: List[BenchmarkResult] = []
    for library_slug, op, dtype in candidates:
        benchmark_name = f"reduce/{op}/{library_slug}/{dtype}"
        if not should_run_benchmark(benchmark_name, filter_pattern):
            continue
        if args.output_format == "table":
            print(f"Benchmarking {benchmark_name}")
        try:
            if library_slug == "numkong":
                all_results.append(
                    benchmark_numkong(
                        op, elements, dtype, args.warmup, args.time_limit, args.seed
                    )
                )
            else:
                all_results.append(
                    benchmark_numpy(
                        op, elements, dtype, args.warmup, args.time_limit, args.seed
                    )
                )
        except Exception as e:
            if args.output_format == "table":
                print(f"  Skipped: {e}")

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "suite": "reduce",
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

    print(f"\nReduction benchmarks: {elements:,} elements")
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
