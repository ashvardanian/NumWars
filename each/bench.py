#!/usr/bin/env python3
"""
Elementwise operation benchmarks: NumKong vs NumPy.

Benchmarks add, scale (alpha * a + beta) and fma (alpha * a * b + beta * c) across multiple data types.

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


INPUT_COUNTS = {"add": 2, "scale": 1, "fma": 3}


def timed_result(
    library: str, operation: str, dtype_str: str, elements: int, func, settings: Settings
) -> BenchmarkResult:
    """Times `func`, which reads `INPUT_COUNTS[operation]` arrays of `elements` and writes one."""
    duration = measure_average_duration(func, settings)
    bytes_processed = (INPUT_COUNTS[operation] + 1) * elements * dtype_itemsize(dtype_str)
    return BenchmarkResult(
        library=library,
        operation=operation,
        input_dtype=dtype_str,
        output_dtype=dtype_str,
        display_signature=display_signature(dtype_str, dtype_str),
        elements=elements,
        duration_secs=duration,
        throughput_gibs=bytes_processed / duration / 2**30,
    )


def benchmark_numkong(operation: str, elements: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a, b, c = (build_array(elements, dtype_str, rng) for _ in range(3))
    if dtype_str == "bf16":
        # NumPy has no bfloat16, so NumKong is told what the buffers hold
        funcs = {
            "add": lambda: nk.add(a, b, a_dtype="bfloat16", b_dtype="bfloat16", out_dtype="bfloat16"),
            "scale": lambda: nk.scale(a, "bfloat16", alpha=2.5, beta=0.5),
            "fma": lambda: nk.fma(a, b, c, "bfloat16", alpha=2.5, beta=0.5),
        }
    else:
        funcs = {
            "add": lambda: nk.add(a, b),
            "scale": lambda: nk.scale(a, alpha=2.5, beta=0.5),
            "fma": lambda: nk.fma(a, b, c, alpha=2.5, beta=0.5),
        }
    return timed_result("NumKong", operation, dtype_str, elements, funcs[operation], settings)


def benchmark_numpy(operation: str, elements: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a, b, c = (build_array(elements, dtype_str, rng) for _ in range(3))
    out, scratch = np.empty_like(a), np.empty_like(a)

    def scale():
        np.multiply(a, a.dtype.type(2.5), out=out, casting="unsafe")
        np.add(out, a.dtype.type(0.5), out=out, casting="unsafe")

    def fma():
        np.multiply(a, b, out=out)
        np.multiply(out, a.dtype.type(2.5), out=out, casting="unsafe")
        np.multiply(c, a.dtype.type(0.5), out=scratch, casting="unsafe")
        np.add(out, scratch, out=out)

    funcs = {"add": lambda: np.add(a, b, out=out), "scale": scale, "fma": fma}
    return timed_result("NumPy", operation, dtype_str, elements, funcs[operation], settings)


candidates = [
    (library, operation, dtype_str)
    for operation in ("add", "scale", "fma")
    for library, dtypes in (("numkong", ("f32", "f64", "f16", "bf16", "i8")), ("numpy", ("f32", "f64", "f16", "i8")))
    for dtype_str in dtypes
]

dispatch = {"numkong": benchmark_numkong, "numpy": benchmark_numpy}


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
    settings = read_settings("each")
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

        all_results.append(dispatch[library_slug](operation, elements, dtype_str, settings))

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
