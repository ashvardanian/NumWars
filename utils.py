"""
Shared utilities for NumWars Python benchmarking scripts.

Common functions for data generation, timing, argument parsing, and results
formatting used across bench_similarity.py, bench_each.py, and bench_dots.py.
"""

import argparse
import os
import re
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

# Configure threading for all numerical libraries based on NUMWARS_THREADS.
# NUMWARS_THREADS=1 (default): single-threaded. NUMWARS_THREADS=0: all cores.
# Must be set before importing numpy or any BLAS-backed library.
_numwars_threads = int(os.environ.get("NUMWARS_THREADS", "1"))
if _numwars_threads == 0:
    _numwars_threads = os.cpu_count() or 1
_threads_str = str(_numwars_threads)

os.environ["OMP_NUM_THREADS"] = _threads_str
os.environ["MKL_NUM_THREADS"] = _threads_str
os.environ["NUMEXPR_NUM_THREADS"] = _threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = _threads_str
os.environ["OPENBLAS_NUM_THREADS"] = _threads_str

import numpy as np

try:
    import ml_dtypes

    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False

T = TypeVar("T")


# region: Environment Variable Helpers
#
# Standardized functions for fetching environment variables consistently.
# Use these instead of raw os.environ.get() calls throughout the codebase.


def get_env(name: str) -> Optional[str]:
    """Get an optional environment variable, returning None if not set."""
    return os.environ.get(name)


def get_env_or_default(name: str, default: str) -> str:
    """Get an environment variable with a default value."""
    return os.environ.get(name, default)


def get_env_parsed(name: str, default: T, parser: Callable[[str], T] = int) -> T:
    """
    Get an environment variable parsed to a type, with a default value.
    Returns the default if the variable is not set or cannot be parsed.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return parser(value)
    except (ValueError, TypeError):
        return default


def get_env_parsed_opt(name: str, parser: Callable[[str], T] = int) -> Optional[T]:
    """
    Get an optional environment variable parsed to a type.
    Returns None if the variable is not set or cannot be parsed.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return parser(value)
    except (ValueError, TypeError):
        return None


def get_env_bool(name: str) -> bool:
    """
    Get a boolean environment variable.
    Accepts "1", "true", or "yes" (case-insensitive) as true values.
    Returns False if not set or set to any other value.
    """
    value = os.environ.get(name, "").lower()
    return value in ("1", "true", "yes")


def get_env_list(name: str) -> List[str]:
    """Parse a comma-separated list of values from an environment variable."""
    value = os.environ.get(name)
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


# endregion

# region: Timing Utilities


def now_ns() -> int:
    """Get current time in nanoseconds for benchmarking."""
    return time.perf_counter_ns()


def measure_latency(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure latency of a function call in seconds.

    Returns:
        (result, duration_secs)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def measure_throughput(
    func: Callable, bytes_processed: int, min_time: float = 1.0
) -> Tuple[float, int]:
    """
    Measure throughput of a function by running it multiple times.

    Args:
        func: Function to benchmark (no arguments)
        bytes_processed: Number of bytes processed per call
        min_time: Minimum time to run benchmarks (seconds)

    Returns:
        (throughput_gbps, iterations)
    """
    iterations = 0
    start = time.perf_counter()

    while True:
        func()
        iterations += 1
        elapsed = time.perf_counter() - start
        if elapsed >= min_time:
            break

    total_bytes = bytes_processed * iterations
    throughput = total_bytes / elapsed / 1e9  # GB/s
    return throughput, iterations


def measure_average_duration(
    func: Callable[[], Any], warmup_seconds: float, profile_seconds: float
) -> float:
    """
    Measure average duration of a function call with warmup and profiling phases.

    Args:
        func: Zero-argument callable to benchmark
        warmup_seconds: Warmup duration in seconds
        profile_seconds: Measurement duration in seconds

    Returns:
        Average duration per call in seconds
    """
    warmup_start = time.perf_counter()
    while (time.perf_counter() - warmup_start) < warmup_seconds:
        func()

    durations = []
    profile_start = time.perf_counter()
    while (time.perf_counter() - profile_start) < profile_seconds:
        iter_start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - iter_start)

    if not durations:
        iter_start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - iter_start)

    return sum(durations) / len(durations)


def normalize_dtype_name(dtype_like) -> str:
    """Normalize a NumPy-style dtype string to a short form (e.g. 'float32' -> 'f32')."""
    text = str(dtype_like).lower()
    mapping = {
        "float64": "f64",
        "float32": "f32",
        "float16": "f16",
        "bfloat16": "bf16",
        "int64": "i64",
        "int32": "i32",
        "int16": "i16",
        "int8": "i8",
        "uint64": "u64",
        "uint32": "u32",
        "uint16": "u16",
        "uint8": "u8",
    }
    return mapping.get(text, text)


def numkong_dtype_name(dtype_str: str) -> str:
    """Map short dtype names (i8, u8, …) to numkong long names (int8, uint8, …)."""
    mapping = {
        "i64": "int64",
        "i32": "int32",
        "i16": "int16",
        "i8": "int8",
        "u64": "uint64",
        "u32": "uint32",
        "u16": "uint16",
        "u8": "uint8",
    }
    return mapping.get(dtype_str, dtype_str)


def calculate_gso_per_sec(num_operations: int, duration_secs: float) -> float:
    """Calculate giga scalar operations per second."""
    return (num_operations / duration_secs) / 1e9 if duration_secs > 0 else 0.0


def calculate_mps(count: int, duration_secs: float) -> float:
    """Calculate millions of pairs (or points) per second."""
    return (count / duration_secs) / 1e6 if duration_secs > 0 else 0.0


# endregion

# region:
# Data Generation
# endregion

# region:


def get_ml_dtype(dtype_str: str):
    """Get ML data type from string (bf16, e4m3, e5m2, i4, u4)."""
    if not HAS_ML_DTYPES:
        raise ImportError(
            "ml_dtypes package is required for exotic types. Install with: pip install ml_dtypes"
        )

    dtype_map = {
        "bf16": ml_dtypes.bfloat16,
        "e4m3": ml_dtypes.float8_e4m3fn,
        "e5m2": ml_dtypes.float8_e5m2,
        "i4": ml_dtypes.int4,
        "u4": ml_dtypes.uint4,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown ML dtype: {dtype_str}. Supported: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def parse_numpy_dtype(dtype_str: str) -> np.dtype:
    """
    Parse a data type string to NumPy dtype.

    Supports: f64, f32, f16, bf16, e4m3, e5m2, i8, i4, u8, u4, complex64, complex128
    """
    # Standard NumPy types
    dtype_map = {
        "f64": np.float64,
        "f32": np.float32,
        "f16": np.float16,
        "i64": np.int64,
        "i32": np.int32,
        "i16": np.int16,
        "i8": np.int8,
        "u64": np.uint64,
        "u32": np.uint32,
        "u16": np.uint16,
        "u8": np.uint8,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }

    if dtype_str in dtype_map:
        return np.dtype(dtype_map[dtype_str])

    # Extended types via ml_dtypes
    if dtype_str in ("bf16", "e4m3", "e5m2", "i4", "u4"):
        return get_ml_dtype(dtype_str)

    raise ValueError(f"Unknown dtype: {dtype_str}")


def generate_random_array(
    shape: Tuple[int, ...], dtype_str: str, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random array with specified shape and data type.

    Args:
        shape: Array shape
        dtype_str: Data type string (e.g., "f32", "f64", "e4m3")
        seed: Random seed for reproducibility

    Returns:
        Random NumPy array
    """
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)

    # Generate in float32 first, then convert
    if dtype_str.startswith("f") or dtype_str in ("bf16", "e4m3", "e5m2"):
        data = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        return data.astype(dtype)
    elif dtype_str.startswith("i"):
        # Signed integers
        info = (
            np.iinfo(dtype)
            if hasattr(np, "iinfo") and dtype_str not in ("i4",)
            else None
        )
        if info:
            return rng.integers(info.min, info.max, size=shape, dtype=dtype)
        else:
            # For i4, generate in int8 range and convert
            return rng.integers(-8, 7, size=shape).astype(dtype)
    elif dtype_str.startswith("u"):
        # Unsigned integers
        info = (
            np.iinfo(dtype)
            if hasattr(np, "iinfo") and dtype_str not in ("u4",)
            else None
        )
        if info:
            return rng.integers(0, info.max, size=shape, dtype=dtype)
        else:
            # For u4, generate in uint8 range and convert
            return rng.integers(0, 15, size=shape).astype(dtype)
    elif dtype_str.startswith("complex"):
        real = rng.uniform(-1.0, 1.0, size=shape)
        imag = rng.uniform(-1.0, 1.0, size=shape)
        return (real + 1j * imag).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype for random generation: {dtype_str}")


def generate_probability_distribution(
    size: int, dtype_str: str = "f32", seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random probability distribution (values sum to 1.0).

    Useful for KL divergence and Jensen-Shannon distance benchmarks.
    """
    rng = np.random.default_rng(seed)
    dtype = parse_numpy_dtype(dtype_str)

    # Generate positive random values
    values = rng.uniform(0.01, 1.0, size=size).astype(np.float32)
    # Normalize to sum to 1
    values = values / values.sum()

    return values.astype(dtype)


# endregion

# region:
# Argument Parsing
# endregion

# region:


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common benchmark arguments to an ArgumentParser.

    Note: dtype filtering is done via --filter regex, not a separate --dtype argument.
    To filter by dtype, use: --filter "f32" or --filter "f32|f64"
    """
    parser.add_argument(
        "-k",
        "--filter",
        metavar="REGEX",
        default=get_env("NUMWARS_FILTER"),
        help="Regex to select which benchmarks to run (or set NUMWARS_FILTER env var). "
        "Examples: --filter 'f32' (only f32), --filter 'angular|dot' (specific metrics)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=get_env_parsed("NUMWARS_WARMUP", 1.0, float),
        help="Warmup time in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=get_env_parsed("NUMWARS_TIME_LIMIT", 5.0, float),
        help="Time limit per benchmark in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )


def should_run_benchmark(
    name: str, filter_pattern: Optional[re.Pattern] = None
) -> bool:
    """
    Check if a benchmark should run based on NUMWARS_FILTER regex.

    This is the ONLY filtering mechanism. All benchmarks generate all combinations
    of dtype × operation/metric by default. Use NUMWARS_FILTER to selectively run
    specific benchmarks via regex matching on the full benchmark name.

    Example benchmark names:
    - "similarity/angular/f32"
    - "each/add/f64"
    - "dots/f32/1024x1024x1024"

    Example filters:
    - NUMWARS_FILTER="f32" → only f32 benchmarks
    - NUMWARS_FILTER="angular|dot" → only angular and dot metrics
    - NUMWARS_FILTER="each/add" → only add operations in each module

    Args:
        name: Benchmark name
        filter_pattern: Compiled regex pattern for name filtering

    Returns:
        True if benchmark should run
    """
    # Check name filter
    if filter_pattern is not None:
        if not filter_pattern.search(name):
            return False

    # No filter set = run all benchmarks
    return True


# endregion

# region:
# Results Formatting
# endregion

# region:


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string (KB, MB, GB, TB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_number(n: int) -> str:
    """Format integer with thousands separators."""
    return f"{n:,}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_results_table(
    results: List[dict], headers: Optional[List[str]] = None
) -> None:
    """
    Print benchmark results as a formatted table.

    Args:
        results: List of result dictionaries
        headers: Optional list of column headers (auto-detected if None)
    """
    try:
        from tabulate import tabulate

        if not results:
            print("No results to display")
            return

        if headers is None:
            headers = list(results[0].keys())

        # Extract data rows
        rows = [[r.get(h, "") for h in headers] for r in results]

        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        # Fallback to simple printing if tabulate not available
        if not results:
            print("No results to display")
            return

        if headers is None:
            headers = list(results[0].keys())

        # Print headers
        print(" | ".join(headers))
        print("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))

        # Print rows
        for r in results:
            print(" | ".join(str(r.get(h, "")) for h in headers))


# endregion

# region:
# region: Dimension Helpers


def get_matrix_dims_width() -> int:
    """
    Get matrix output width in C = A @ B.T where C is height×width.
    Used in dots/ module for matrix multiplication benchmarks.
    """
    return get_env_parsed("NUMWARS_DIMS_WIDTH", 2048, int)


def get_matrix_dims_height() -> int:
    """
    Get matrix output height in C = A @ B.T where C is height×width.
    Used in dots/ module for matrix multiplication benchmarks.
    """
    return get_env_parsed("NUMWARS_DIMS_HEIGHT", 2048, int)


def get_matrix_dims_depth() -> int:
    """
    Get matrix shared dimension in A @ B.T where A is height×depth and B is width×depth.
    Used in dots/ module for matrix multiplication benchmarks.
    """
    return get_env_parsed("NUMWARS_DIMS_DEPTH", 2048, int)


def get_tensor_dims() -> int:
    """
    Get tensor dimensions for elementwise operations.
    Used in each/ module for elementwise operation benchmarks.
    """
    return get_env_parsed("NUMWARS_DIMS", 1_000_000, int)


def get_vector_dims() -> int:
    """
    Get vector dimensions for similarity operations.
    Used in similarity/ module for pairwise vector similarity benchmarks.
    """
    return get_env_parsed("NUMWARS_DIMS", 2048, int)


def get_batch_size() -> int:
    """Alias for get_vector_dims() — kept for backward compatibility."""
    return get_vector_dims()


def get_thread_count() -> int:
    """
    Get thread count for parallel benchmarks.
    Defaults to 1 (single-threaded). Set to 0 to use all logical CPUs.
    """
    n = get_env_parsed("NUMWARS_THREADS", 1, int)
    if n == 0:
        return os.cpu_count() or 1
    return n


# endregion

# region:
# Benchmark Helpers
# endregion

# region:


def calculate_throughput(num_bytes: int, duration_secs: float) -> float:
    """Calculate throughput in GB/s."""
    if duration_secs > 0:
        return num_bytes / duration_secs / 1e9
    return 0.0


def calculate_ops_per_sec(num_operations: int, duration_secs: float) -> float:
    """
    Calculate operations per second.

    Args:
        num_operations: Total number of operations
        duration_secs: Duration in seconds

    Returns:
        Operations per second
    """
    return num_operations / duration_secs if duration_secs > 0 else 0.0


def format_ops_per_sec(ops_per_sec: float) -> str:
    """
    Format scalar operations per second with auto-scaling units.

    Similar to StringWars CUPS (Characters Used Per Second) formatter.
    Automatically selects appropriate scale.

    Args:
        ops_per_sec: Operations per second

    Returns:
        Formatted string like "345.23 GSO/s"

    Examples:
        >>> format_ops_per_sec(1234567890)
        "1.23 GSO/s"
        >>> format_ops_per_sec(2.5e12)
        "2.50 TSO/s"
    """
    if ops_per_sec >= 1e12:
        return f"{ops_per_sec / 1e12:.2f} TSO/s"
    elif ops_per_sec >= 1e9:
        return f"{ops_per_sec / 1e9:.2f} GSO/s"
    elif ops_per_sec >= 1e6:
        return f"{ops_per_sec / 1e6:.2f} MSO/s"
    elif ops_per_sec >= 1e3:
        return f"{ops_per_sec / 1e3:.2f} KSO/s"
    else:
        return f"{ops_per_sec:.2f} SO/s"


def benchmark_with_time_limit(
    func: Callable[[], Any],
    warmup_seconds: float,
    profile_seconds: float,
) -> Tuple[float, int]:
    """
    Run benchmark until time limit is reached.

    Args:
        func: Function to benchmark (no arguments)
        warmup_seconds: Warmup duration
        profile_seconds: Measurement duration

    Returns:
        (average_duration, iteration_count)
    """
    # Warmup phase
    warmup_start = time.perf_counter()
    warmup_iters = 0
    while (time.perf_counter() - warmup_start) < warmup_seconds:
        func()
        warmup_iters += 1

    # Measurement phase
    durations = []
    measure_start = time.perf_counter()
    while (time.perf_counter() - measure_start) < profile_seconds:
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        durations.append(end - start)

    if not durations:
        # If profile_seconds is very short, run at least once
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        durations.append(end - start)

    return sum(durations) / len(durations), len(durations)
