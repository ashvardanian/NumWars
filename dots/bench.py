#!/usr/bin/env python3
"""
Packed GEMM-style benchmarks: NumKong vs NumPy.

Computes the GEMM C = A × Bᵀ using NumKong packed matrices and mainstream Python baselines, and the
SYRK C = A × Aᵀ, where NumKong fills the upper triangle and NumPy hands `a @ a.T` to BLAS `syrk`.
The SYRK rows count only the n(n+1)/2 pairs of one triangle times the depth.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,ml_dtypes dots/bench.py

Or with traditional pip:
    pip install -e ".[dots]"
    python dots/bench.py
"""

import json
import sys
from dataclasses import dataclass

import numpy as np

from numwars import (
    Settings,
    calculate_gso_per_sec,
    format_duration,
    measure_average_duration,
    normalize_dtype_name,
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
    throughput_gibs: float
    workload: str = "dots"


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


def build_matrix(shape: tuple[int, int], dtype_str: str, rng: np.random.Generator) -> np.ndarray:
    dtype = parse_numpy_dtype(dtype_str)
    data = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return data.astype(dtype)


def benchmark_numkong(height: int, width: int, depth: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a = build_matrix((height, depth), dtype_str, rng)
    b = build_matrix((width, depth), dtype_str, rng)
    packed_b = nk.dots_pack(b, dtype=numkong_dtype_name(dtype_str))
    sample_output = nk.dots_packed(a, packed_b)
    output_dtype = normalize_dtype_name(getattr(sample_output, "dtype", dtype_str))
    output_array = np.empty((height, width), dtype=parse_numpy_dtype(output_dtype))
    output_tensor = nk.Tensor(output_array, dtype=numkong_dtype_name(output_dtype))

    duration = measure_average_duration(
        lambda: nk.dots_packed(a, packed_b, out=output_tensor, threads=settings.threads),
        settings,
    )
    num_operations = 2 * height * width * depth
    out_itemsize = dtype_itemsize(output_dtype)
    bytes_processed = (height * depth + width * depth) * a.itemsize + (height * width) * out_itemsize
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
        throughput_gibs=bytes_processed / duration / 2**30,
    )


def benchmark_numpy(height: int, width: int, depth: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a = build_matrix((height, depth), dtype_str, rng)
    b = build_matrix((width, depth), dtype_str, rng)
    out = np.empty((height, width), dtype=a.dtype)
    duration = measure_average_duration(lambda: np.matmul(a, b.T, out=out), settings)
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
        throughput_gibs=bytes_processed / duration / 2**30,
    )


def symmetric_operations(count: int, depth: int) -> int:
    """Credits only the n(n+1)/2 pairs of one triangle, at a multiply and an add per dimension."""
    return 2 * (count * (count + 1) // 2) * depth


def benchmark_numkong_symmetric(count: int, depth: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a = build_matrix((count, depth), dtype_str, rng)
    dtype = numkong_dtype_name(dtype_str)
    sample_output = nk.dots_symmetric(a, dtype=dtype)
    output_dtype = normalize_dtype_name(getattr(sample_output, "dtype", dtype_str))
    output_array = np.empty((count, count), dtype=parse_numpy_dtype(output_dtype))
    output_tensor = nk.Tensor(output_array, dtype=numkong_dtype_name(output_dtype))

    duration = measure_average_duration(
        lambda: nk.dots_symmetric(a, dtype=dtype, out=output_tensor, threads=settings.threads),
        settings,
    )
    out_itemsize = dtype_itemsize(output_dtype)
    bytes_processed = count * depth * a.itemsize + (count * (count + 1) // 2) * out_itemsize
    return BenchmarkResult(
        library="NumKong",
        input_dtype=dtype_str,
        output_dtype=output_dtype,
        display_signature=display_signature_name(dtype_str, output_dtype),
        height=count,
        width=count,
        depth=depth,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(symmetric_operations(count, depth), duration),
        throughput_gibs=bytes_processed / duration / 2**30,
        workload="symmetric",
    )


def benchmark_numpy_symmetric(count: int, depth: int, dtype_str: str, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    a = build_matrix((count, depth), dtype_str, rng)
    out = np.empty((count, count), dtype=a.dtype)
    duration = measure_average_duration(lambda: np.matmul(a, a.T, out=out), settings)
    bytes_processed = (count * depth + count * (count + 1) // 2) * a.itemsize
    return BenchmarkResult(
        library="NumPy",
        input_dtype=dtype_str,
        output_dtype=dtype_str,
        display_signature=display_signature_name(dtype_str, dtype_str),
        height=count,
        width=count,
        depth=depth,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(symmetric_operations(count, depth), duration),
        throughput_gibs=bytes_processed / duration / 2**30,
        workload="symmetric",
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    if result.workload == "symmetric":
        shape = f"{result.height}x{result.depth}"
        benchmark_id = f"dots/symmetric/{result.library.lower()}/{result.input_dtype}/{shape}"
    else:
        shape = f"{result.height}x{result.width}x{result.depth}"
        benchmark_id = f"dots/{result.library.lower()}/{result.input_dtype}/{shape}"
    return {
        "suite": "dots",
        "workload": result.workload,
        "benchmark_id": benchmark_id,
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
        "throughput_gibs": result.throughput_gibs,
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
        ("dots", "numkong", "i8"),
        ("symmetric", "numkong", "i8"),
        ("dots", "numkong", "f16"),
        ("symmetric", "numkong", "f16"),
        ("dots", "numkong", "bf16"),
        ("symmetric", "numkong", "bf16"),
        ("dots", "numkong", "f32"),
        ("symmetric", "numkong", "f32"),
        ("dots", "numpy", "f32"),
        ("symmetric", "numpy", "f32"),
    ]

    all_results: list[BenchmarkResult] = []
    for workload, library_slug, dtype in candidates:
        if workload == "symmetric":
            benchmark_name = f"dots/symmetric/{library_slug}/{dtype}/{height}x{depth}"
        else:
            benchmark_name = f"dots/{library_slug}/{dtype}/{height}x{width}x{depth}"
        if not settings.selects(benchmark_name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {benchmark_name}")
        if workload == "symmetric":
            benchmark = benchmark_numkong_symmetric if library_slug == "numkong" else benchmark_numpy_symmetric
            all_results.append(benchmark(height, depth, dtype, settings))
        elif library_slug == "numkong":
            all_results.append(benchmark_numkong(height, width, depth, dtype, settings))
        else:
            all_results.append(benchmark_numpy(height, width, depth, dtype, settings))

    if settings.output == "json":
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

    print(f"GEMM dots: A({height}×{depth}) × Bᵀ({depth}×{width}) → C({height}×{width})")
    print(f"SYRK dots: A({height}×{depth}) × Aᵀ({depth}×{height}) → C({height}×{height}), counting one triangle")
    print()
    table_rows = [
        {
            "Library": result.library,
            "Kernel": "SYRK" if result.workload == "symmetric" else "GEMM",
            "Precision": result.display_signature,
            "GSO/s": f"{result.gso_per_sec:.2f}",
            "GB/s": f"{result.throughput_gibs:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
