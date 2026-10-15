#!/usr/bin/env python3
"""
Ragged scaled-dot-product attention benchmarks: NumKong vs PyTorch vs NumPy.

Computes O = softmax(Q K^T / sqrt(d)) V for a single ragged segment with 8 heads,
comparing NumKong's pre-packed KV-cache kernels against PyTorch's fused CPU
`scaled_dot_product_attention` and a NumPy matmul + softmax baseline.

With a CUDA build of NumKong and PyTorch's, each GPU in NUMWARS_DEVICES adds rows named like
`attention/numkong/bf16/16384x128x16384<cuda:0>` for every dtype NumKong's CUDA attention takes,
against PyTorch's fused `scaled_dot_product_attention` under each backend that accepts the shape,
labelled like `<cudnn,cuda:0>`, and a CuPy matmul + softmax floor mirroring the NumPy one. Every
round synchronizes, and a first round that strays from NumKong's CPU attention on the leading
queries is marked ✗.

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
    GpuDevice,
    Settings,
    calculate_gso_per_sec,
    format_duration,
    gpu_check,
    gpu_devices,
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

try:
    import torch
    import torch.nn.functional as F
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
    device: str = ""
    mismatch: str = ""


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
    packed = nk.attention_pack(keys, values, key_offsets=kv_offsets, depth=head_dim, threads=settings.threads)

    duration = measure_average_duration(
        lambda: nk.attention_packed(queries, packed, query_offsets=query_offsets, threads=settings.threads),
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
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype_str]
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


# Relative error a GPU row may show against the CPU, which sums the keys in another order.
GPU_TOLERANCE = 1e-3

# Relative error NumKong's GPU rows may show against its CPU attention per input dtype: the CPU rounds
# `bf16` and `e4m3` probabilities to BF16, so two roundings that fall differently still agree.
NUMKONG_TOLERANCES = {"bf16": 2**-6, "f16": GPU_TOLERANCE, "e4m3": 2**-6, "i8": GPU_TOLERANCE}

# Leading queries a GPU result's first round is compared on.
CHECKED_ROWS = 16

# NumKong's CUDA attention dtypes and the PyTorch type holding their bits.
GPU_DTYPES = {"bf16": "bfloat16", "f16": "float16", "e4m3": "float8_e4m3fn", "i8": "int8"}

# PyTorch's fused attention backends on CUDA, by the name their rows carry.
SDPA_BACKENDS = {"cudnn": "CUDNN_ATTENTION", "flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION"}


def gpu_operands(device: GpuDevice, rows: int, width: int, dtype_str: str, rng: np.random.Generator):
    """Random `[rows, width]` operands as a host array NumKong's CPU kernels read, and the same bits in a
    PyTorch tensor on `device`: floats normal with a 0.3 deviation and `i8` within ±32, as the CPU rows draw."""
    import torch

    if dtype_str == "i8":
        host = rng.integers(-32, 32, (rows, width)).astype(np.int8)
    else:
        host = (rng.standard_normal((rows, width)) * 0.3).astype(np.float32).astype(parse_numpy_dtype(dtype_str))
    bits = torch.from_numpy(host.view(f"u{host.itemsize}"))
    return host, bits.to(f"cuda:{device.ordinal}").view(getattr(torch, GPU_DTYPES[dtype_str]))


def pinned_offsets(count: int) -> np.ndarray:
    """The offsets `[0, count]` of one segment, in pinned memory the host and a GPU both read."""
    import torch

    return torch.tensor([0, count], dtype=torch.int32).pin_memory().numpy().view(np.uint32)


def gpu_result(library: str, dtype_str: str, output_dtype: str, shape: tuple, duration: float, label: str, mismatch):
    kv_length, head_dim, query_count = shape
    return BenchmarkResult(
        library=library,
        input_dtype=dtype_str,
        output_dtype=output_dtype,
        display_signature=display_signature_name(dtype_str, output_dtype),
        kv_length=kv_length,
        head_dim=head_dim,
        query_count=query_count,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(attention_operations(*shape), duration),
        device=label,
        mismatch="✗" if mismatch else "",
    )


def benchmark_gpu(
    device: GpuDevice, kv_length: int, head_dim: int, query_count: int, dtype_str: str, settings: Settings
) -> list[BenchmarkResult]:
    """NumKong's attention on `device` and PyTorch's fused backends, each first round compared with
    NumKong's CPU attention on the leading queries, printing and marking any mismatch."""
    import torch
    import torch.nn.attention as sdpa

    shape = (kv_length, head_dim, query_count)
    size = f"{kv_length}x{head_dim}x{query_count}"
    names = {"numkong": f"attention/numkong/{dtype_str}/{size}{device.label}"}
    if dtype_str in ("bf16", "f16"):
        for backend in SDPA_BACKENDS:
            names[backend] = f"attention/torch/{dtype_str}/{size}<{backend},cuda:{device.ordinal}>"
    if not any(map(settings.selects, names.values())):
        return []
    rng = np.random.default_rng(settings.seed)
    width = HEAD_COUNT * head_dim
    (queries_host, queries), (keys_host, keys), (values_host, values) = (
        gpu_operands(device, rows, width, dtype_str, rng) for rows in (query_count, kv_length, kv_length)
    )
    leading = min(CHECKED_ROWS, query_count)
    reference = nk.attention_pack(
        keys_host, values_host, key_offsets=np.array([0, kv_length], np.uint32), depth=head_dim
    )
    expected = np.asarray(
        nk.attention_packed(queries_host[:leading], reference, query_offsets=np.array([0, leading], np.uint32))
    )

    nk_dtype = numkong_dtype_name(dtype_str)
    pack_bytes = nk.AttentionPackedMatrix.pack_size(
        HEAD_COUNT, head_dim, kv_length, 1, nk_dtype, capabilities=int(device.capabilities)
    )
    storage = torch.empty(pack_bytes, dtype=torch.uint8, device=queries.device)
    key_offsets, query_offsets = pinned_offsets(kv_length), pinned_offsets(query_count)
    key_values = nk.attention_pack(
        nk.from_dlpack(keys),
        nk.from_dlpack(values),
        key_offsets=key_offsets,
        depth=head_dim,
        out=nk.from_dlpack(storage),
        stream=device.stream,
    )
    output = torch.empty((query_count, width), dtype=torch.float32, device=queries.device)
    queries_tensor, output_tensor = nk.from_dlpack(queries), nk.from_dlpack(output)

    def numkong_round():
        nk.attention_packed(
            queries_tensor, key_values, query_offsets=query_offsets, out=output_tensor, stream=device.stream
        )
        nk.synchronize(device.capabilities, stream=device.stream)
        return output

    rounds = {"numkong": (numkong_round, "f32", NUMKONG_TOLERANCES[dtype_str])}
    if dtype_str in ("bf16", "f16"):
        heads_first = [
            tensor.view(-1, HEAD_COUNT, head_dim).transpose(0, 1).contiguous()[None]
            for tensor in (queries, keys, values)
        ]
        # The rivals round their outputs to their input precision, coarser than the keys' summation error.
        tolerance = max(GPU_TOLERANCE, 2 * torch.finfo(queries.dtype).eps)
        for backend, member in SDPA_BACKENDS.items():

            def torch_round(member=member):
                with sdpa.sdpa_kernel(getattr(sdpa.SDPBackend, member)), torch.no_grad():
                    attended = torch.nn.functional.scaled_dot_product_attention(*heads_first)
                torch.cuda.synchronize(queries.device)
                return attended

            rounds[backend] = (torch_round, dtype_str, tolerance)

    results = []
    for library, (one_round, output_dtype, tolerance) in rounds.items():
        name = names[library]
        if not settings.selects(name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {name}")
        try:
            found = one_round()
        except RuntimeError as error:  # a backend that declines this shape or device skips its own row
            print(f"Skipping {name}: {str(error).splitlines()[0]}")
            continue
        if found.dim() == 4:  # PyTorch's heads-first [1, heads, queries, depth] back to [queries, heads × depth]
            found = found[0].transpose(0, 1).reshape(query_count, width)
        mismatch = gpu_check(name, found[:leading].double().cpu().numpy(), expected, False, tolerance)
        if mismatch:
            print(f"✗ {mismatch}")
        duration = measure_average_duration(one_round, settings)
        library_name, label = (
            ("NumKong", device.label)
            if library == "numkong"
            else ("PyTorch", names[library][names[library].index("<") :])
        )
        results.append(gpu_result(library_name, dtype_str, output_dtype, shape, duration, label, mismatch))
    torch.cuda.empty_cache()
    return results


def benchmark_cupy_gpu(
    device: GpuDevice, kv_length: int, head_dim: int, query_count: int, settings: Settings
) -> list[BenchmarkResult]:
    """The NumPy matmul + softmax composition on `device` through CuPy in F32, its first round compared
    with NumPy on the leading queries."""
    name = f"attention/cupy/f32/{kv_length}x{head_dim}x{query_count}{device.label}"
    if not settings.selects(name):
        return []
    try:
        import cupy
    except ImportError:
        return []
    rng = np.random.default_rng(settings.seed)
    queries_host = (rng.standard_normal((HEAD_COUNT, query_count, head_dim)) * 0.3).astype(np.float32)
    keys_host = (rng.standard_normal((HEAD_COUNT, kv_length, head_dim)) * 0.3).astype(np.float32)
    values_host = (rng.standard_normal((HEAD_COUNT, kv_length, head_dim)) * 0.3).astype(np.float32)
    scale = 1.0 / np.sqrt(head_dim)
    leading = min(CHECKED_ROWS, query_count)
    scores = np.matmul(queries_host[:, :leading], keys_host.transpose(0, 2, 1)) * scale
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = np.matmul(weights / weights.sum(axis=-1, keepdims=True), values_host).transpose(1, 0, 2)
    with cupy.cuda.Device(device.ordinal):
        queries, keys, values = (cupy.asarray(array) for array in (queries_host, keys_host, values_host))

        def cupy_round():
            scores = cupy.matmul(queries, keys.transpose(0, 2, 1)) * scale
            scores -= scores.max(axis=-1, keepdims=True)
            weights = cupy.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True)
            attended = cupy.matmul(weights, values)
            cupy.cuda.Device().synchronize()
            return attended

        if settings.output == "table":
            print(f"Benchmarking {name}")
        found = cupy_round()[:, :leading].get().transpose(1, 0, 2)
        mismatch = gpu_check(name, found.reshape(leading, -1), expected.reshape(leading, -1), False, GPU_TOLERANCE)
        if mismatch:
            print(f"✗ {mismatch}")
        duration = measure_average_duration(cupy_round, settings)
        cupy.get_default_memory_pool().free_all_blocks()
    shape = (kv_length, head_dim, query_count)
    return [gpu_result("CuPy", "f32", "f32", shape, duration, device.label, mismatch)]


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "attention",
        "workload": "attention",
        "benchmark_id": f"attention/{result.library.lower()}/{result.input_dtype}/"
        f"{result.kv_length}x{result.head_dim}x{result.query_count}{result.device}",
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
    settings = read_settings("attention")
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
        ("numkong", "f16"),
        ("numkong", "e4m3"),
        ("numkong", "i8"),
        ("torch", "f32"),
        ("torch", "bf16"),
        ("torch", "f16"),
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

    for device in gpu_devices(settings):
        for dtype in GPU_DTYPES:
            all_results.extend(benchmark_gpu(device, kv_length, head_dim, query_count, dtype, settings))
        all_results.extend(benchmark_cupy_gpu(device, kv_length, head_dim, query_count, settings))

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

    print(f"Attention: {HEAD_COUNT} heads, Q({query_count}×{head_dim}) over KV({kv_length}×{head_dim})")
    print()
    table_rows = [
        {
            "Library": f"{result.library}{result.device}{result.mismatch}",
            "Precision": result.display_signature,
            "GSO/s": f"{result.gso_per_sec:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
