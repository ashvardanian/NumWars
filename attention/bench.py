#!/usr/bin/env python3
"""
Ragged scaled-dot-product attention benchmarks: NumKong vs PyTorch vs NumPy.

Computes O = softmax(Q K^T / sqrt(d)) V for a single ragged segment with 8 heads,
comparing NumKong's pre-packed KV-cache kernels against PyTorch's fused CPU
`scaled_dot_product_attention` and a NumPy matmul + softmax baseline.

Can be run with uv:
    uv run --with numkong,numpy,tabulate,torch attention/bench.py

Or with traditional pip:
    pip install -e ".[attention]"
    python attention/bench.py
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
    print_results_table,
    print_settings,
    read_settings,
)

try:
    import numkong as nk
except ImportError:
    print("Error: numkong not found. Install with: pip install numkong")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
except ImportError:
    torch = None

HEAD_COUNT = 8


@dataclass
class BenchmarkResult:
    library: str
    input_dtype: str
    output_dtype: str
    display_signature: str
    kv_length: int
    head_dim: int
    query_count: int
    duration_secs: float
    gso_per_sec: float


def display_signature_name(input_dtype: str, output_dtype: str) -> str:
    return f"{input_dtype} → {output_dtype}"


def attention_operations(kv_length: int, head_dim: int, query_count: int) -> int:
    # Q·K^T and P·V each cost 2·h·q·kv·d scalar operations
    return 4 * HEAD_COUNT * query_count * kv_length * head_dim


def benchmark_numkong(
    kv_length: int, head_dim: int, query_count: int, dtype_str: str, settings: Settings
) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    row_width = HEAD_COUNT * head_dim
    if dtype_str == "i8":
        queries = nk.Tensor(rng.integers(-32, 32, (query_count, row_width)).astype(np.int8))
        keys = nk.Tensor(rng.integers(-32, 32, (kv_length, row_width)).astype(np.int8))
        values = nk.Tensor(rng.integers(-32, 32, (kv_length, row_width)).astype(np.int8))
    else:
        queries = nk.Tensor((rng.standard_normal((query_count, row_width)) * 0.3).astype(np.float32)).astype(dtype_str)
        keys = nk.Tensor((rng.standard_normal((kv_length, row_width)) * 0.3).astype(np.float32)).astype(dtype_str)
        values = nk.Tensor((rng.standard_normal((kv_length, row_width)) * 0.3).astype(np.float32)).astype(dtype_str)
    kv_offsets = np.array([0, kv_length], dtype=np.uint32)
    query_offsets = np.array([0, query_count], dtype=np.uint32)
    packed = nk.attention_pack(keys, values, segment_offsets=kv_offsets, head_dim=head_dim, threads=1)

    duration = measure_average_duration(
        lambda: nk.attention_packed(queries, packed, query_offsets=query_offsets, threads=1),
        settings,
    )
    num_operations = attention_operations(kv_length, head_dim, query_count)
    return BenchmarkResult(
        library="NumKong",
        input_dtype=dtype_str,
        output_dtype="f32",
        display_signature=display_signature_name(dtype_str, "f32"),
        kv_length=kv_length,
        head_dim=head_dim,
        query_count=query_count,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


def benchmark_torch(
    kv_length: int, head_dim: int, query_count: int, dtype_str: str, settings: Settings
) -> BenchmarkResult:
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[dtype_str]
    generator = torch.Generator().manual_seed(settings.seed)
    queries = torch.randn(1, HEAD_COUNT, query_count, head_dim, generator=generator).to(dtype) * 0.3
    keys = torch.randn(1, HEAD_COUNT, kv_length, head_dim, generator=generator).to(dtype) * 0.3
    values = torch.randn(1, HEAD_COUNT, kv_length, head_dim, generator=generator).to(dtype) * 0.3

    def _torch_attention():
        with torch.no_grad():
            return F.scaled_dot_product_attention(queries, keys, values)

    duration = measure_average_duration(_torch_attention, settings)
    num_operations = attention_operations(kv_length, head_dim, query_count)
    return BenchmarkResult(
        library="PyTorch",
        input_dtype=dtype_str,
        output_dtype=dtype_str,
        display_signature=display_signature_name(dtype_str, dtype_str),
        kv_length=kv_length,
        head_dim=head_dim,
        query_count=query_count,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


def benchmark_numpy(kv_length: int, head_dim: int, query_count: int, settings: Settings) -> BenchmarkResult:
    rng = np.random.default_rng(settings.seed)
    queries = (rng.standard_normal((HEAD_COUNT, query_count, head_dim)) * 0.3).astype(np.float32)
    keys = (rng.standard_normal((HEAD_COUNT, kv_length, head_dim)) * 0.3).astype(np.float32)
    values = (rng.standard_normal((HEAD_COUNT, kv_length, head_dim)) * 0.3).astype(np.float32)
    scale = 1.0 / np.sqrt(head_dim)

    def _numpy_attention():
        scores = np.matmul(queries, keys.transpose(0, 2, 1)) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        return np.matmul(weights, values)

    duration = measure_average_duration(_numpy_attention, settings)
    num_operations = attention_operations(kv_length, head_dim, query_count)
    return BenchmarkResult(
        library="NumPy",
        input_dtype="f32",
        output_dtype="f32",
        display_signature=display_signature_name("f32", "f32"),
        kv_length=kv_length,
        head_dim=head_dim,
        query_count=query_count,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "attention",
        "workload": "attention",
        "benchmark_id": f"attention/{result.library.lower()}/{result.input_dtype}/"
        f"{result.kv_length}x{result.head_dim}x{result.query_count}",
        "library": result.library,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_dtype": result.display_signature,
        "display_signature": result.display_signature,
        "accumulator_dtype": "f32",
        "height": result.kv_length,
        "width": result.head_dim,
        "depth": result.query_count,
        "primary_value": result.gso_per_sec,
        "unit": "GSO/s",
        "duration_secs": result.duration_secs,
    }


def main():
    settings = read_settings()
    if settings.output == "table":
        print_settings(settings)

    kv_length, head_dim, query_count = settings.dims_height, settings.dims_width, settings.dims_depth

    metadata = {
        "kv_length": kv_length,
        "head_dim": head_dim,
        "query_count": query_count,
        "head_count": HEAD_COUNT,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
        "torch_version": getattr(torch, "__version__", None) if torch else None,
    }

    candidates = [
        ("numkong", "bf16"),
        ("numkong", "e4m3"),
        ("numkong", "i8"),
        ("torch", "f32"),
        ("torch", "bf16"),
        ("numpy", "f32"),
    ]

    all_results: list[BenchmarkResult] = []
    for library_slug, dtype in candidates:
        benchmark_name = f"attention/{library_slug}/{dtype}/{kv_length}x{head_dim}x{query_count}"
        if not settings.selects(benchmark_name):
            continue
        if library_slug == "torch" and torch is None:
            continue
        if settings.output == "table":
            print(f"Benchmarking {benchmark_name}")
        if library_slug == "numkong":
            all_results.append(benchmark_numkong(kv_length, head_dim, query_count, dtype, settings))
        elif library_slug == "torch":
            all_results.append(benchmark_torch(kv_length, head_dim, query_count, dtype, settings))
        else:
            all_results.append(benchmark_numpy(kv_length, head_dim, query_count, settings))

    if settings.output == "json":
        print(
            json.dumps(
                {
                    "suite": "attention",
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

    print(f"Attention: {HEAD_COUNT} heads, Q({query_count}x{head_dim}) over KV({kv_length}x{head_dim})")
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
