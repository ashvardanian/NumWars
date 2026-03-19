#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packed GEMM-style benchmarks: NumKong vs NumPy.

Computes C = A @ B.T using NumKong packed matrices and mainstream Python baselines.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,ml_dtypes dots/bench.py

Or with traditional pip:
    pip install -e ".[dots]"
    python dots/bench.py

Environment variables:
    NUMWARS_FILTER - Regex filter for benchmark names
    NUMWARS_DIMS_WIDTH - Matrix C width (default: 2048)
    NUMWARS_DIMS_HEIGHT - Matrix C height (default: 2048)
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
        calculate_gso_per_sec,
        format_duration,
        get_matrix_dims_depth,
        get_matrix_dims_height,
        get_matrix_dims_width,
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
    input_dtype: str
    output_dtype: str
    display_signature: str
    height: int
    width: int
    depth: int
    duration_secs: float
    gso_per_sec: float
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



def build_matrix(shape: tuple[int, int], dtype_str: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)
    data = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return data.astype(dtype)


def benchmark_numkong(
    height: int, width: int, depth: int, dtype_str: str, warmup: float, profile: float
) -> BenchmarkResult:
    a = build_matrix((height, depth), dtype_str, 42)
    b = build_matrix((width, depth), dtype_str, 43)
    packed_b = nk.dots_pack(b, dtype=numkong_dtype_name(dtype_str))
    sample_output = nk.dots_packed(a, packed_b)
    output_dtype = normalize_dtype_name(getattr(sample_output, "dtype", dtype_str))
    output_array = np.empty((height, width), dtype=parse_numpy_dtype(output_dtype))
    output_tensor = nk.Tensor(output_array, dtype=numkong_dtype_name(output_dtype))

    duration = measure_average_duration(
        lambda: nk.dots_packed(a, packed_b, out=output_tensor),
        warmup,
        profile,
    )
    num_operations = 2 * height * width * depth
    out_itemsize = dtype_itemsize(output_dtype)
    bytes_processed = (height * depth + width * depth) * a.itemsize + (
        height * width
    ) * out_itemsize
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
        throughput_gbs=bytes_processed / duration / 1e9,
    )


def benchmark_numpy(
    height: int, width: int, depth: int, dtype_str: str, warmup: float, profile: float
) -> BenchmarkResult:
    a = build_matrix((height, depth), dtype_str, 42)
    b = build_matrix((width, depth), dtype_str, 43)
    out = np.empty((height, width), dtype=a.dtype)
    duration = measure_average_duration(
        lambda: np.matmul(a, b.T, out=out), warmup, profile
    )
    num_operations = 2 * height * width * depth
    bytes_processed = (height * depth + width * depth + height * width) * a.itemsize
    return BenchmarkResult(
        library="NumPy",
        input_dtype=dtype_str,
        output_dtype=dtype_str,
        display_signature=display_signature_name(dtype_str, dtype_str),
        height=height,
        width=width,
        depth=depth,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
        throughput_gbs=bytes_processed / duration / 1e9,
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "dots",
        "workload": "dots",
        "benchmark_id": f"dots/{result.library.lower()}/{result.input_dtype}/{result.height}x{result.width}x{result.depth}",
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
        "throughput_gbs": result.throughput_gbs,
        "duration_secs": result.duration_secs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark packed GEMM-style dot products",
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
        ("numkong", "i8"),
        ("numkong", "f16"),
        ("numkong", "bf16"),
        ("numkong", "f32"),
        ("numpy", "f32"),
    ]

    all_results: List[BenchmarkResult] = []
    for library_slug, dtype in candidates:
        benchmark_name = f"dots/{library_slug}/{dtype}/{height}x{width}x{depth}"
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
                benchmark_numpy(
                    height, width, depth, dtype, args.warmup, args.time_limit
                )
            )

    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "suite": "dots",
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

    print(
        f"Packed GEMM-style dots: A({height}x{depth}) @ B.T({width}x{depth}) -> ({height}x{width})"
    )
    print()
    table_rows = [
        {
            "Library": result.library,
            "Precision": result.display_signature,
            "GSO/s": f"{result.gso_per_sec:.2f}",
            "GB/s": f"{result.throughput_gbs:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
