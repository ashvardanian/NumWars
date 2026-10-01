#!/usr/bin/env python3
"""
Reduction benchmarks: NumKong vs NumPy.

Covers flat-vector sum and row-wise L2 norms.

Can be run with uv:
    uv run --with numkong,numpy,tabulate reduce/bench.py

Or with traditional pip:
    pip install -e ".[reduce]"
    python reduce/bench.py
"""

import json
import sys
from dataclasses import dataclass

import numpy as np

from numwars import (
    Settings,
    format_duration,
    measure_average_duration,
    parse_numpy_dtype,
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
    library: str
    operation: str
    input_dtype: str
    output_dtype: str
    display_signature: str
    elements: int
    duration_secs: float
    throughput_gibs: float


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


def build_vector(elements: int, dtype_str: str, rng: np.random.Generator) -> np.ndarray:
    dtype = parse_numpy_dtype(dtype_str)
    if dtype_str.startswith(("i", "u")):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=elements, dtype=dtype)
    data = rng.uniform(-1.0, 1.0, size=elements).astype(np.float32)
    return data.astype(dtype)


def build_matrix(batch_size: int, ndim: int, dtype_str: str, rng: np.random.Generator) -> np.ndarray:
    dtype = parse_numpy_dtype(dtype_str)
    data = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(np.float32)
    return data.astype(dtype)


def benchmark_numpy(op: str, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    if op == "norm":
        batch_size, ndim = settings.batch_size, settings.dims
        actual_elements = batch_size * ndim
        matrix = build_matrix(batch_size, ndim, dtype_str, rng)
        func = lambda: np.linalg.norm(matrix, axis=1)
        bytes_processed = actual_elements * matrix.itemsize
    else:
        actual_elements = settings.batch_size
        x = build_vector(actual_elements, dtype_str, rng)
        if op == "sum":
            func = lambda: np.sum(x)
        else:
            raise ValueError(f"Unknown operation: {op}")
        bytes_processed = actual_elements * x.itemsize

    duration = measure_average_duration(func, settings)
    throughput_gibs = bytes_processed / duration / 2**30 if duration > 0 else 0.0
    output_dtype = dtype_str
    if op == "norm":
        output_dtype = "f64"

    return BenchmarkResult(
        library="NumPy",
        operation=op,
        input_dtype=dtype_str,
        output_dtype=output_dtype,
        display_signature=display_signature_name(dtype_str, output_dtype),
        elements=actual_elements,
        duration_secs=duration,
        throughput_gibs=throughput_gibs,
    )


def benchmark_numkong(op: str, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    if op == "norm":
        batch_size, ndim = settings.batch_size, settings.dims
        actual_elements = batch_size * ndim
        matrix = build_matrix(batch_size, ndim, dtype_str, rng)
        t = nk.Tensor(matrix)
        func = lambda: nk.norm(t, axis=1)
        bytes_processed = actual_elements * matrix.itemsize
    else:
        actual_elements = settings.batch_size
        x = build_vector(actual_elements, dtype_str, rng)
        if op == "sum":
            t = nk.Tensor(x)
            func = lambda: nk.sum(t)
        else:
            raise ValueError(f"Unknown operation: {op}")
        bytes_processed = actual_elements * x.itemsize

    duration = measure_average_duration(func, settings)
    throughput_gibs = bytes_processed / duration / 2**30 if duration > 0 else 0.0
    output_dtype = dtype_str
    if op == "norm":
        output_dtype = "f64"

    return BenchmarkResult(
        library="NumKong",
        operation=op,
        input_dtype=dtype_str,
        output_dtype=output_dtype,
        display_signature=display_signature_name(dtype_str, output_dtype),
        elements=actual_elements,
        duration_secs=duration,
        throughput_gibs=throughput_gibs,
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
        "primary_value": result.throughput_gibs,
        "unit": "GB/s",
        "throughput_gibs": result.throughput_gibs,
        "duration_secs": result.duration_secs,
    }


def main():
    settings = read_settings()
    if settings.output == "table":
        print_settings(settings)

    metadata = {
        "batch_size": settings.batch_size,
        "dims": settings.dims,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
    }

    candidates = [
        ("numpy", "sum", "f32"),
        ("numpy", "sum", "f64"),
        ("numpy", "sum", "i8"),
        ("numpy", "sum", "u8"),
        ("numkong", "sum", "f32"),
        ("numkong", "sum", "f64"),
        ("numkong", "sum", "i8"),
        ("numkong", "sum", "u8"),
        ("numkong", "sum", "bf16"),
        ("numpy", "norm", "f32"),
        ("numpy", "norm", "f64"),
        ("numkong", "norm", "f32"),
        ("numkong", "norm", "f64"),
        ("numkong", "norm", "bf16"),
    ]

    all_results: list[BenchmarkResult] = []
    for library_slug, op, dtype in candidates:
        benchmark_name = f"reduce/{op}/{library_slug}/{dtype}"
        if not settings.selects(benchmark_name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {benchmark_name}")
        try:
            if library_slug == "numkong":
                all_results.append(benchmark_numkong(op, dtype, settings))
            else:
                all_results.append(benchmark_numpy(op, dtype, settings))
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            if settings.output == "table":
                print(f"  Skipped: {e}")

    if settings.output == "json":
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

    print(f"\nReduction benchmarks: {settings.batch_size:,} elements, or rows of {settings.dims}")
    print()
    table_rows = [
        {
            "Library": result.library,
            "Operation": result.operation,
            "Precision": result.display_signature,
            "GB/s": f"{result.throughput_gibs:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
