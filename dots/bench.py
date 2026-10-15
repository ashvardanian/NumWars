#!/usr/bin/env python3
"""
Packed GEMM-style benchmarks: NumKong vs NumPy.

Computes the GEMM C = A × Bᵀ using NumKong packed matrices and mainstream Python baselines, and the
SYRK C = A × Aᵀ, where NumKong fills the upper triangle and NumPy hands `a @ a.T` to BLAS `syrk`.
The SYRK rows count only the n(n+1)/2 pairs of one triangle times the depth.

With a CUDA build of NumKong and PyTorch's, each GPU in NUMWARS_DEVICES adds rows named like
`dots/numkong/bf16/16384x16384x16384<cuda:0>` for every dtype NumKong's CUDA kernels take through
DLPack, against `torch.matmul`, `torch._scaled_mm` and `torch._int_mm`, and CuPy's `matmul`. Their
operands are random PyTorch tensors, every round synchronizes, and each first round is checked
against NumKong's CPU kernel on the leading rows.

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
    GpuDevice,
    Settings,
    calculate_gso_per_sec,
    format_duration,
    gpu_check,
    gpu_devices,
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
    device: str = ""


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


# Relative error a GPU row may show against the CPU, which sums the depth in another order.
GPU_TOLERANCE = 1e-3

# The libraries with GPU rows, by the slug their row names carry.
GPU_LIBRARIES = {"numkong": "NumKong", "pytorch": "PyTorch", "cupy": "CuPy"}

# Leading rows of a GPU result its first round is checked on.
CHECKED_ROWS = 16

# NumKong's CUDA dtypes and the PyTorch type holding their bits. The packed `e2m1`, `i4` and `u4` hold
# two dimensions per byte. FP6 is missing, as no Python library hands it to DLPack on a GPU.
GPU_DTYPES = {
    "f64": "float64",
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "e4m3": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
    "e2m1": "uint8",
    "i8": "int8",
    "u8": "uint8",
    "i4": "uint8",
    "u4": "uint8",
}


def gpu_operands(device: GpuDevice, rows: int, depth: int, dtype_str: str, seed: int):
    """Random `[rows, depth]` operands as a host array NumKong's CPU kernels read, and the same bits in
    a PyTorch tensor on `device`: floats in [-1, 1), `i8` within ±100, `u8` up to 200, any packed nibbles."""
    import torch

    rng = np.random.default_rng(seed)
    if GPU_DTYPES[dtype_str] == "uint8" and dtype_str != "u8":
        host = rng.integers(0, 256, size=(rows, depth // 2), dtype=np.uint8)
    elif dtype_str in ("i8", "u8"):
        host = rng.integers(*((-100, 101) if dtype_str == "i8" else (0, 201)), size=(rows, depth))
        host = host.astype(parse_numpy_dtype(dtype_str))
    else:
        host = build_matrix((rows, depth), dtype_str, rng)
    bits = torch.from_numpy(host.view(f"u{host.itemsize}"))
    return host, bits.to(f"cuda:{device.ordinal}").view(getattr(torch, GPU_DTYPES[dtype_str]))


def cpu_leading_rows(a_host: np.ndarray, b_host: np.ndarray, dtype_str: str) -> np.ndarray:
    """NumKong's CPU result for the leading rows of `a_host` against every row of `b_host`."""
    packed = nk.dots_pack(b_host, dtype=numkong_dtype_name(dtype_str))
    return np.asarray(nk.dots_packed(a_host[:CHECKED_ROWS], packed))


def gpu_rivals(dtype_str: str, a, b) -> dict:
    """The PyTorch and CuPy rounds of `C = A × Bᵀ` for `dtype_str`, by library, each with the output it
    writes and synchronizes. cuBLASLt multiplies E5M2 only against E4M3, and neither library packs nibbles."""
    import torch

    rounds = {}

    def output(dtype):
        return torch.empty((a.shape[0], b.shape[0]), dtype=dtype, device=a.device)

    if dtype_str in ("f64", "f32", "f16", "bf16"):
        product = output(a.dtype)
        rounds["pytorch"] = (lambda: torch.matmul(a, b.T, out=product), product)
    elif dtype_str == "e4m3":
        scale, product = torch.ones((), dtype=torch.float32, device=a.device), output(torch.float32)
        rounds["pytorch"] = (
            lambda: torch._scaled_mm(a, b.T, scale, scale, out_dtype=torch.float32, out=product),
            product,
        )
    elif dtype_str == "i8":
        product = output(torch.int32)
        rounds["pytorch"] = (lambda: torch._int_mm(a, b.T, out=product), product)
    try:
        import cupy
    except ImportError:
        cupy = None
    if cupy is not None and dtype_str in ("f64", "f32", "f16"):
        cupy_product = output(a.dtype)
        left, right, target = cupy.from_dlpack(a), cupy.from_dlpack(b).T, cupy.from_dlpack(cupy_product)
        rounds["cupy"] = (lambda: cupy.matmul(left, right, out=target), cupy_product)
    return {
        library: ((lambda multiply=multiply: (multiply(), torch.cuda.synchronize(a.device))), product)
        for library, (multiply, product) in rounds.items()
    }


def benchmark_gpu(
    device: GpuDevice, height: int, width: int, depth: int, dtype_str: str, workload: str, settings: Settings
) -> list[BenchmarkResult]:
    """NumKong's GEMM or SYRK on `device` and its rivals, each first round checked against NumKong's CPU kernel."""
    import torch

    symmetric = workload == "symmetric"
    if symmetric:
        width = height
        shape, operations = f"{height}x{depth}", symmetric_operations(height, depth)
    else:
        shape, operations = f"{height}x{width}x{depth}", 2 * height * width * depth
    prefix = "symmetric/" if symmetric else ""
    names = {library: f"dots/{prefix}{library}/{dtype_str}/{shape}{device.label}" for library in GPU_LIBRARIES}
    if not any(map(settings.selects, names.values())):
        return []
    a_host, a = gpu_operands(device, height, depth, dtype_str, settings.seed)
    b_host, b = (a_host, a) if symmetric else gpu_operands(device, width, depth, dtype_str, settings.seed + 1)
    expected = cpu_leading_rows(a_host, b_host, dtype_str)
    nk_dtype = numkong_dtype_name(dtype_str)
    output = torch.empty((height, width), dtype=getattr(torch, str(expected.dtype)), device=a.device)
    a_tensor, output_tensor = nk.from_dlpack(a), nk.from_dlpack(output)
    if symmetric:

        def numkong_round() -> None:
            nk.dots_symmetric(a_tensor, dtype=nk_dtype, out=output_tensor, stream=device.stream)
            nk.synchronize(device.capabilities, stream=device.stream)

    else:
        pack_bytes = nk.PackedMatrix.pack_size(width, depth, nk_dtype, capabilities=device.capabilities)
        storage = torch.empty(pack_bytes, dtype=torch.uint8, device=a.device)
        packed = nk.dots_pack(nk.from_dlpack(b), dtype=nk_dtype, out=nk.from_dlpack(storage), stream=device.stream)

        def numkong_round() -> None:
            nk.dots_packed(a_tensor, packed, out=output_tensor, stream=device.stream)
            nk.synchronize(device.capabilities, stream=device.stream)

    rounds = {"numkong": (numkong_round, output), **gpu_rivals(dtype_str, a, b)}
    results = []
    for library, (one_round, round_output) in rounds.items():
        name = names[library]
        if not settings.selects(name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {name}")
        one_round()
        found = round_output[:CHECKED_ROWS].double().cpu().numpy()
        # Rivals round their outputs to their own precision, which may be coarser than the depth's error.
        tolerance = max(
            GPU_TOLERANCE, 2 * torch.finfo(round_output.dtype).eps if round_output.is_floating_point() else 0
        )
        mismatch = gpu_check(name, found, expected, symmetric, tolerance)
        if mismatch and library == "numkong":
            raise SystemExit(mismatch)
        if mismatch:
            print(f"Skipping a rival that {mismatch}")
            continue
        duration = measure_average_duration(one_round, settings)
        bytes_processed = a_host.nbytes + (0 if symmetric else b_host.nbytes) + round_output.nbytes
        round_dtype = normalize_dtype_name(str(round_output.dtype).removeprefix("torch."))
        results.append(
            BenchmarkResult(
                library=GPU_LIBRARIES[library],
                input_dtype=dtype_str,
                output_dtype=round_dtype,
                display_signature=display_signature_name(dtype_str, round_dtype),
                height=height,
                width=width,
                depth=depth,
                duration_secs=duration,
                gso_per_sec=calculate_gso_per_sec(operations, duration),
                throughput_gibs=bytes_processed / duration / 2**30,
                workload=workload,
                device=device.label,
            )
        )
    torch.cuda.empty_cache()
    return results


def result_to_entry(result: BenchmarkResult) -> dict:
    if result.workload == "symmetric":
        shape = f"{result.height}x{result.depth}"
        benchmark_id = f"dots/symmetric/{result.library.lower()}/{result.input_dtype}/{shape}{result.device}"
    else:
        shape = f"{result.height}x{result.width}x{result.depth}"
        benchmark_id = f"dots/{result.library.lower()}/{result.input_dtype}/{shape}{result.device}"
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
    settings = read_settings("dots")
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

    for device in gpu_devices(settings):
        for dtype in GPU_DTYPES:
            for workload in ("dots", "symmetric"):
                all_results.extend(benchmark_gpu(device, height, width, depth, dtype, workload, settings))

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
            "Library": f"{result.library}{result.device}",
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
