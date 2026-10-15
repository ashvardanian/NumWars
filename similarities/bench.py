#!/usr/bin/env python3
"""
All-pairs similarity matrix benchmarks: NumKong vs SciPy.

Computes pairwise distance matrices using packed kernels (angulars_packed / euclideans_packed) for NumKong and cdist for SciPy. Both libraries produce (batch_size, batch_size) output matrices.

The SYRK rows score one matrix against itself, A × Aᵀ, with the `*_symmetric` kernels filling the upper triangle for NumKong and `pdist` returning the condensed upper triangle for SciPy, and count only the n(n+1)/2 pairs of that triangle times the depth.

With a CUDA build of NumKong and PyTorch's, each GPU in NUMWARS_DEVICES adds rows named like `similarities/angular/numkong/bf16<cuda:0>` for every dtype NumKong's CUDA kernels take through DLPack. CuPy races them the way the CPU rivals do, a cuBLAS product and one fused norm epilogue over norms computed beforehand, and `torch.cdist` races the Euclidean ones. Operands are random PyTorch tensors, every round synchronizes, and each first round is checked against NumKong's CPU kernel on the leading rows.

Can be run with uv:
    uv run --with numkong,numpy,scipy,tabulate similarities/bench.py

Or with traditional pip:
    pip install -e ".[similarities]"
    python similarities/bench.py
"""

import json
import sys
from dataclasses import dataclass

import numpy as np
import scipy
import scipy.spatial.distance as spd

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
    metric: str
    input_dtype: str
    output_dtype: str
    display_signature: str
    ndim: int
    batch_size: int
    duration_secs: float
    gso_per_sec: float
    device: str = ""


def display_signature_name(input_dtype: str, output_dtype: str) -> str:
    return f"{input_dtype} \u2192 {output_dtype}"


def build_matrices(batch_size: int, ndim: int, dtype_str: str, seed: int):
    """Build two random matrices of shape (batch_size, ndim) in the given dtype."""
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)
    if dtype_str.startswith("f") or dtype_str in ("bf16",):
        A = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(np.float32).astype(dtype)
        B = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(np.float32).astype(dtype)
    elif dtype_str.startswith("i"):
        A = rng.integers(-100, 100, size=(batch_size, ndim), dtype=dtype)
        B = rng.integers(-100, 100, size=(batch_size, ndim), dtype=dtype)
    elif dtype_str.startswith("u"):
        A = rng.integers(0, 200, size=(batch_size, ndim), dtype=dtype)
        B = rng.integers(0, 200, size=(batch_size, ndim), dtype=dtype)
    else:
        A = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(dtype)
        B = rng.uniform(-1.0, 1.0, size=(batch_size, ndim)).astype(dtype)
    return A, B


def build_bits(batch_size: int, ndim: int, seed: int) -> np.ndarray:
    """Build a random (batch_size, ndim) matrix of zeros and ones, one byte per bit."""
    return np.random.default_rng(seed).integers(0, 2, size=(batch_size, ndim), dtype=np.uint8)


def symmetric_operations(batch_size: int, ndim: int) -> int:
    """Credits only the n(n+1)/2 pairs of one triangle, times the depth."""
    return batch_size * (batch_size + 1) // 2 * ndim


# Map from our metric names to scipy metric names
SCIPY_METRIC_MAP = {
    "angular": "cosine",
    "euclidean": "euclidean",
    "hamming": "hamming",
    "jaccard": "jaccard",
}

SYMMETRIC_KERNELS = {
    "angular": nk.angulars_symmetric,
    "euclidean": nk.euclideans_symmetric,
    "hamming": nk.hammings_symmetric,
    "jaccard": nk.jaccards_symmetric,
}


def benchmark_scipy(
    metric: str,
    batch_size: int,
    ndim: int,
    settings: Settings,
) -> BenchmarkResult:
    A, B = build_matrices(batch_size, ndim, "f32", settings.seed)
    out = np.empty((batch_size, batch_size), dtype=np.float64)
    scipy_metric = SCIPY_METRIC_MAP[metric]

    duration = measure_average_duration(
        lambda: spd.cdist(A, B, scipy_metric, out=out),
        settings,
    )
    num_operations = batch_size * batch_size * ndim
    return BenchmarkResult(
        library="SciPy",
        metric=metric,
        input_dtype="f32",
        output_dtype="f64",
        display_signature=display_signature_name("f32", "f64"),
        ndim=ndim,
        batch_size=batch_size,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


PACK_DTYPE_MAP = {"i8": "int8", "u8": "uint8"}


def benchmark_numkong(
    metric: str,
    batch_size: int,
    ndim: int,
    dtype_str: str,
    settings: Settings,
) -> BenchmarkResult:
    A, B = build_matrices(batch_size, ndim, dtype_str, settings.seed)

    # Pack B once (maps short dtype names to the ones dots_pack expects)
    pack_dtype = PACK_DTYPE_MAP.get(dtype_str, dtype_str)
    B_packed = nk.dots_pack(B, dtype=pack_dtype)

    # Pick the correct packed function
    func = nk.angulars_packed if metric == "angular" else nk.euclideans_packed

    # Discover output dtype from a warm-up call
    sample = func(A, B_packed)
    out_dtype = normalize_dtype_name(getattr(sample, "dtype", "f32"))
    out_array = np.empty((batch_size, batch_size), dtype=parse_numpy_dtype(out_dtype))
    out_tensor = nk.Tensor(out_array, dtype=numkong_dtype_name(out_dtype))

    duration = measure_average_duration(
        lambda: func(A, B_packed, out=out_tensor, threads=settings.threads),
        settings,
    )
    num_operations = batch_size * batch_size * ndim
    return BenchmarkResult(
        library="NumKong",
        metric=metric,
        input_dtype=dtype_str,
        output_dtype=out_dtype,
        display_signature=display_signature_name(dtype_str, out_dtype),
        ndim=ndim,
        batch_size=batch_size,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(num_operations, duration),
    )


def benchmark_scipy_symmetric(metric: str, batch_size: int, ndim: int, settings: Settings) -> BenchmarkResult:
    if metric in ("hamming", "jaccard"):
        vectors, input_dtype = build_bits(batch_size, ndim, settings.seed).astype(bool), "bool"
    else:
        vectors, input_dtype = build_matrices(batch_size, ndim, "f32", settings.seed)[0], "f32"
    out = np.empty(batch_size * (batch_size - 1) // 2, dtype=np.float64)
    scipy_metric = SCIPY_METRIC_MAP[metric]

    duration = measure_average_duration(
        lambda: spd.pdist(vectors, scipy_metric, out=out),
        settings,
    )
    return BenchmarkResult(
        library="SciPy",
        metric=f"symmetric/{metric}",
        input_dtype=input_dtype,
        output_dtype="f64",
        display_signature=display_signature_name(input_dtype, "f64"),
        ndim=ndim,
        batch_size=batch_size,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(symmetric_operations(batch_size, ndim), duration),
    )


def benchmark_numkong_symmetric(
    metric: str,
    batch_size: int,
    ndim: int,
    dtype_str: str,
    settings: Settings,
) -> BenchmarkResult:
    if metric in ("hamming", "jaccard"):
        vectors, dtype = np.packbits(build_bits(batch_size, ndim, settings.seed), axis=1), "uint1"
    else:
        vectors, dtype = build_matrices(batch_size, ndim, dtype_str, settings.seed)[0], numkong_dtype_name(dtype_str)
    func = SYMMETRIC_KERNELS[metric]

    # Discover output dtype from a warm-up call
    sample = func(vectors, dtype=dtype)
    out_dtype = normalize_dtype_name(getattr(sample, "dtype", "f32"))
    out_array = np.empty((batch_size, batch_size), dtype=parse_numpy_dtype(out_dtype))
    out_tensor = nk.Tensor(out_array, dtype=numkong_dtype_name(out_dtype))

    duration = measure_average_duration(
        lambda: func(vectors, dtype=dtype, out=out_tensor, threads=settings.threads),
        settings,
    )
    return BenchmarkResult(
        library="NumKong",
        metric=f"symmetric/{metric}",
        input_dtype=dtype_str,
        output_dtype=out_dtype,
        display_signature=display_signature_name(dtype_str, out_dtype),
        ndim=ndim,
        batch_size=batch_size,
        duration_secs=duration,
        gso_per_sec=calculate_gso_per_sec(symmetric_operations(batch_size, ndim), duration),
    )


# Relative error a GPU row may show against the CPU, which sums the depth in another order.
GPU_TOLERANCE = 1e-3

# Leading rows of a GPU result its first round is checked on.
CHECKED_ROWS = 16

# The libraries with GPU rows, by the slug their row names carry.
GPU_LIBRARIES = {"numkong": "NumKong", "cupy": "CuPy", "pytorch": "PyTorch"}

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

PACKED_KERNELS = {"angular": nk.angulars_packed, "euclidean": nk.euclideans_packed}

# The norm epilogues CuPy fuses after its cuBLAS product, reading norms computed beforehand.
EPILOGUES = {
    "angular": ("T dot, F inverse_a, F inverse_b", "F distance", "distance = 1 - (F)dot * inverse_a * inverse_b"),
    "euclidean": (
        "T dot, F squared_a, F squared_b",
        "F distance",
        "distance = sqrt(max(squared_a + squared_b - 2 * (F)dot, (F)0))",
    ),
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
        host = rng.uniform(-1.0, 1.0, size=(rows, depth)).astype(np.float32).astype(parse_numpy_dtype(dtype_str))
    bits = torch.from_numpy(host.view(f"u{host.itemsize}"))
    return host, bits.to(f"cuda:{device.ordinal}").view(getattr(torch, GPU_DTYPES[dtype_str]))


def gpu_rivals(metric: str, dtype_str: str, a, b) -> dict:
    """The CuPy and PyTorch rounds of `metric` between the rows of `a` and `b`, by library, each returning
    its synchronized output: CuPy's cuBLAS product and fused epilogue, and `torch.cdist` for Euclidean."""
    import torch

    rounds = {}
    try:
        import cupy
    except ImportError:
        cupy = None
    if cupy is not None and dtype_str in ("f64", "f32", "f16", "bf16"):
        left, right = cupy.from_dlpack(a), cupy.from_dlpack(b)
        precision = cupy.float64 if dtype_str == "f64" else cupy.float32
        squared_a = (left.astype(precision) ** 2).sum(axis=1)
        squared_b = (right.astype(precision) ** 2).sum(axis=1)
        if metric == "angular":
            squared_a, squared_b = 1 / cupy.sqrt(squared_a), 1 / cupy.sqrt(squared_b)
        norms = squared_a[:, None], squared_b[None, :]
        product = cupy.empty((left.shape[0], right.shape[0]), dtype=left.dtype)
        distances = cupy.empty(product.shape, dtype=precision)
        epilogue = cupy.ElementwiseKernel(*EPILOGUES[metric], name=f"numwars_{metric}s")

        def cupy_round():
            epilogue(cupy.matmul(left, right.T, out=product), *norms, distances)
            torch.cuda.synchronize(a.device)
            return torch.from_dlpack(distances)

        rounds["cupy"] = cupy_round
    if metric == "euclidean" and dtype_str in ("f64", "f32"):

        def pytorch_round():
            distances = torch.cdist(a, b)
            torch.cuda.synchronize(a.device)
            return distances

        rounds["pytorch"] = pytorch_round
    return rounds


def benchmark_gpu(
    device: GpuDevice, workload: str, batch_size: int, ndim: int, dtype_str: str, settings: Settings
) -> list[BenchmarkResult]:
    """NumKong's `workload` on `device` and its rivals, each first round checked against NumKong's CPU kernel."""
    import torch

    metric = workload.removeprefix("symmetric/")
    symmetric = workload != metric
    names = {library: f"similarities/{workload}/{library}/{dtype_str}{device.label}" for library in GPU_LIBRARIES}
    if not any(map(settings.selects, names.values())):
        return []
    a_host, a = gpu_operands(device, batch_size, ndim, dtype_str, settings.seed)
    b_host, b = (a_host, a) if symmetric else gpu_operands(device, batch_size, ndim, dtype_str, settings.seed + 1)
    nk_dtype = numkong_dtype_name(dtype_str)
    expected = np.asarray(PACKED_KERNELS[metric](a_host[:CHECKED_ROWS], nk.dots_pack(b_host, dtype=nk_dtype)))
    output = torch.empty((batch_size, batch_size), dtype=getattr(torch, str(expected.dtype)), device=a.device)
    a_tensor, output_tensor = nk.from_dlpack(a), nk.from_dlpack(output)
    if symmetric:
        kernel = SYMMETRIC_KERNELS[metric]

        def numkong_round():
            kernel(a_tensor, dtype=nk_dtype, out=output_tensor, stream=device.stream)
            nk.synchronize(device.capabilities, stream=device.stream)
            return output

    else:
        pack_bytes = nk.PackedMatrix.pack_size(batch_size, ndim, nk_dtype, capabilities=device.capabilities)
        storage = torch.empty(pack_bytes, dtype=torch.uint8, device=a.device)
        packed = nk.dots_pack(nk.from_dlpack(b), dtype=nk_dtype, out=nk.from_dlpack(storage), stream=device.stream)
        kernel = PACKED_KERNELS[metric]

        def numkong_round():
            kernel(a_tensor, packed, out=output_tensor, stream=device.stream)
            nk.synchronize(device.capabilities, stream=device.stream)
            return output

    operations = symmetric_operations(batch_size, ndim) if symmetric else batch_size * batch_size * ndim
    results = []
    for library, one_round in {"numkong": numkong_round, **gpu_rivals(metric, dtype_str, a, b)}.items():
        name = names[library]
        if not settings.selects(name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {name}")
        found = one_round()
        # Rivals keep the product in their input precision, which may be coarser than the depth's error.
        tolerance = max(GPU_TOLERANCE, 2 * torch.finfo(a.dtype).eps if library != "numkong" else 0)
        mismatch = gpu_check(name, found[:CHECKED_ROWS].double().cpu().numpy(), expected, symmetric, tolerance)
        if mismatch and library == "numkong":
            raise SystemExit(mismatch)
        if mismatch:
            print(f"Skipping a rival that {mismatch}")
            continue
        duration = measure_average_duration(one_round, settings)
        output_dtype = normalize_dtype_name(str(found.dtype).removeprefix("torch."))
        results.append(
            BenchmarkResult(
                library=GPU_LIBRARIES[library],
                metric=workload,
                input_dtype=dtype_str,
                output_dtype=output_dtype,
                display_signature=display_signature_name(dtype_str, output_dtype),
                ndim=ndim,
                batch_size=batch_size,
                duration_secs=duration,
                gso_per_sec=calculate_gso_per_sec(operations, duration),
                device=device.label,
            )
        )
    torch.cuda.empty_cache()
    return results


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "similarities",
        "workload": result.metric,
        "benchmark_id": f"similarities/{result.metric}/{result.library.lower()}/{result.input_dtype}{result.device}",
        "library": result.library,
        "metric": result.metric.removeprefix("symmetric/"),
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_dtype": result.display_signature,
        "display_signature": result.display_signature,
        "ndim": result.ndim,
        "batch_size": result.batch_size,
        "primary_value": result.gso_per_sec,
        "unit": "GSO/s",
        "duration_secs": result.duration_secs,
    }


def main():
    settings = read_settings("similarities")
    if settings.output == "table":
        print_settings(settings)

    ndim, batch_size = settings.dims, settings.batch_size

    metadata = {
        "ndim": ndim,
        "batch_size": batch_size,
        "numkong_version": getattr(nk, "__version__", None),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }

    candidates = [
        ("numkong", "angular", "f32"),
        ("numkong", "symmetric/angular", "f32"),
        ("numkong", "angular", "bf16"),
        ("numkong", "symmetric/angular", "bf16"),
        ("numkong", "angular", "i8"),
        ("numkong", "symmetric/angular", "i8"),
        ("numkong", "angular", "u8"),
        ("numkong", "symmetric/angular", "u8"),
        ("scipy", "angular", "f32"),
        ("scipy", "symmetric/angular", "f32"),
        ("numkong", "euclidean", "f32"),
        ("numkong", "symmetric/euclidean", "f32"),
        ("numkong", "euclidean", "bf16"),
        ("numkong", "symmetric/euclidean", "bf16"),
        ("numkong", "euclidean", "i8"),
        ("numkong", "symmetric/euclidean", "i8"),
        ("numkong", "euclidean", "u8"),
        ("numkong", "symmetric/euclidean", "u8"),
        ("scipy", "euclidean", "f32"),
        ("scipy", "symmetric/euclidean", "f32"),
        ("numkong", "symmetric/hamming", "u1"),
        ("scipy", "symmetric/hamming", "bool"),
        ("numkong", "symmetric/jaccard", "u1"),
        ("scipy", "symmetric/jaccard", "bool"),
    ]

    all_results: list[BenchmarkResult] = []
    for library_slug, workload, dtype in candidates:
        benchmark_name = f"similarities/{workload}/{library_slug}/{dtype}"
        if not settings.selects(benchmark_name):
            continue
        if settings.output == "table":
            print(f"Benchmarking {benchmark_name}")
        metric = workload.removeprefix("symmetric/")
        try:
            if workload != metric and library_slug == "numkong":
                all_results.append(benchmark_numkong_symmetric(metric, batch_size, ndim, dtype, settings))
            elif workload != metric:
                all_results.append(benchmark_scipy_symmetric(metric, batch_size, ndim, settings))
            elif library_slug == "numkong":
                all_results.append(benchmark_numkong(metric, batch_size, ndim, dtype, settings))
            else:
                all_results.append(benchmark_scipy(metric, batch_size, ndim, settings))
        except (AttributeError, TypeError, ValueError) as e:
            if settings.output == "table":
                print(f"  Error: {e}")

    for device in gpu_devices(settings):
        for workload in ("angular", "symmetric/angular", "euclidean", "symmetric/euclidean"):
            for dtype in GPU_DTYPES:
                all_results.extend(benchmark_gpu(device, workload, batch_size, ndim, dtype, settings))

    if settings.output == "json":
        print(
            json.dumps(
                {
                    "suite": "similarities",
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
    print(f"GEMM similarity: A({batch_size}×{ndim}) × Bᵀ({ndim}×{batch_size}) → C({batch_size}×{batch_size})")
    print(
        f"SYRK similarity: A({batch_size}×{ndim}) × Aᵀ({ndim}×{batch_size}) → C({batch_size}×{batch_size}), "
        "counting one triangle"
    )
    print()
    table_rows = [
        {
            "Library": f"{result.library}{result.device}",
            "Kernel": "SYRK" if result.metric.startswith("symmetric/") else "GEMM",
            "Metric": result.metric.removeprefix("symmetric/"),
            "Precision": result.display_signature,
            "GSO/s": f"{result.gso_per_sec:.2f}",
            "Time": format_duration(result.duration_secs),
        }
        for result in all_results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
