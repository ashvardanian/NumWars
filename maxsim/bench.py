#!/usr/bin/env python3
"""
MaxSim (ColBERT-style late-interaction) benchmarks: NumKong vs NumPy.

Computes score = Sigma_i max_j dot(q_i, d_j) using NumKong's maxsim API
and a NumPy matmul + row-max + sum baseline.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,ml_dtypes maxsim/bench.py

Or with traditional pip:
    pip install -e ".[maxsim]"
    python maxsim/bench.py
"""

import json
import sys
from dataclasses import dataclass

import numpy as np

# Import numwars first: it sets the thread counts that must be in place before NumPy loads OpenBLAS.
from numwars import (
    Settings,
    calculate_gso_per_sec,
    format_duration,
    measure_average_duration,
    numkong_dtype_name,
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


def build_matrix(shape: tuple[int, int], dtype_str: str, rng: np.random.Generator) -> np.ndarray:
    dtype = parse_numpy_dtype(dtype_str)
    data = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return data.astype(dtype)


def benchmark_numkong(height: int, width: int, depth: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    queries = build_matrix((height, depth), dtype_str, rng)
    documents = build_matrix((width, depth), dtype_str, rng)
    nk_dtype = numkong_dtype_name(dtype_str)
    packed_queries = nk.maxsim_pack(queries, dtype=nk_dtype)
    packed_documents = nk.maxsim_pack(documents, dtype=nk_dtype)
    output_dtype = ACCUMULATOR_DTYPE.get(dtype_str, dtype_str)

    duration = measure_average_duration(
        lambda: nk.maxsim_packed(packed_queries, packed_documents),
        settings,
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


def benchmark_numpy(height: int, width: int, depth: int, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    queries = build_matrix((height, depth), "f32", rng)
    documents = build_matrix((width, depth), "f32", rng)
    out = np.empty((height, width), dtype=np.float32)

    def _numpy_maxsim():
        np.matmul(queries, documents.T, out=out)
        return out.max(axis=1).sum()

    duration = measure_average_duration(_numpy_maxsim, settings)
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
    settings = read_settings()
    if settings.output == "table":
        print_settings(settings)

    height, width, depth = settings.dims_height, settings.dims_width, settings.dims_depth

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

    all_results: list[BenchmarkResult] = []
    for library_slug, dtype in candidates:
        benchmark_name = f"maxsim/{library_slug}/{dtype}/{height}x{width}x{depth}"
        if not settings.selects(benchmark_name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {benchmark_name}")
        if library_slug == "numkong":
            all_results.append(benchmark_numkong(height, width, depth, dtype, settings))
        else:
            all_results.append(benchmark_numpy(height, width, depth, settings))

    if settings.output == "json":
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
