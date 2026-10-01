#!/usr/bin/env python3
"""
Elementwise operation benchmarks: NumKong vs NumPy.

Benchmarks add operations across multiple data types.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,ml_dtypes each/bench.py

Or with traditional pip:
    pip install -e ".[each]"
    python each/bench.py
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
    throughput_gibs: float


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


def build_array(elements: int, dtype_str: str, rng: np.random.Generator) -> np.ndarray:
    if dtype_str == "bf16":
        return rng.uniform(-1.0, 1.0, size=(elements,)).astype(np.float32).astype(parse_numpy_dtype("bf16"))
    np_dtype = parse_numpy_dtype(dtype_str)
    if dtype_str == "i8":
        return rng.integers(-128, 127, size=(elements,), dtype=np_dtype)
    return rng.uniform(-1.0, 1.0, size=(elements,)).astype(np.float32).astype(np_dtype)


def benchmark_numkong_add(elements: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a = build_array(elements, dtype_str, rng)
    b = build_array(elements, dtype_str, rng)

    if dtype_str == "bf16":
        func = lambda: nk.add(a, b, a_dtype="bfloat16", b_dtype="bfloat16", out_dtype="bfloat16")
    else:
        func = lambda: nk.add(a, b)

    duration = measure_average_duration(func, settings)
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
        throughput_gibs=bytes_processed / duration / 2**30,
    )


def benchmark_numpy_add(elements: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a = build_array(elements, dtype_str, rng)
    b = build_array(elements, dtype_str, rng)
    out = np.empty_like(a)

    func = lambda: np.add(a, b, out=out)

    duration = measure_average_duration(func, settings)
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
        throughput_gibs=bytes_processed / duration / 2**30,
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
        "primary_value": result.throughput_gibs,
        "unit": "GB/s",
        "throughput_gibs": result.throughput_gibs,
        "duration_secs": result.duration_secs,
    }


def main():
    settings = read_settings()
    if settings.output == "table":
        print_settings(settings)

    elements = settings.batch_size

    metadata = {
        "elements": elements,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
    }

    all_results: list[BenchmarkResult] = []
    for library_slug, operation, dtype_str in candidates:
        benchmark_name = f"each/{operation}/{dtype_str}"
        if not settings.selects(benchmark_name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {benchmark_name} ({library_slug})")

        func = dispatch[(library_slug, operation)]
        result = func(elements, dtype_str, settings)
        all_results.append(result)

    if settings.output == "json":
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
            "GB/s": f"{result.throughput_gibs:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
