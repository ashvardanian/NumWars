"""Shared harness for the NumWars Python suites. Mirrors `numwars.rs`.

Variable                Default  Meaning
NUMWARS_FILTER          none     Regex over benchmark names, or a substring if not a regex
NUMWARS_SEED            42       32-bit seed for random inputs, or `random`
NUMWARS_WARMUP          1s       Warm-up per benchmark, like `200ms` or `1s`
NUMWARS_TIME_LIMIT      10s      Measurement time per benchmark, like `200ms` or `10s`
NUMWARS_BATCH_PER_CORE  2048     Items per call, times the threads: elements, rows, points
NUMWARS_THREADS         1        Threads for NumKong and competitors, `0` for all cores
NUMWARS_DIMS            1536     Vector length
NUMWARS_DIMS_HEIGHT     1024     Rows of `C = A @ B.T` in GEMM-shaped suites
NUMWARS_DIMS_WIDTH      128      Columns of `C = A @ B.T` in GEMM-shaped suites
NUMWARS_DIMS_DEPTH      1536     Shared dimension of `A` and `B` in GEMM-shaped suites
NUMWARS_OUTPUT          table    Human-readable `table` or machine-readable `json`
NUMWARS_SCIPY           false    Whether the `similarity` suite also times SciPy
"""

import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

# region: Environment variables


def env_text(name: str) -> str | None:
    """Reads `name`, or `None` when it is unset or empty."""
    return os.environ.get(name) or None


def env_parsed(name: str, fallback: T, parse: Callable[[str], T | None], expected: str) -> T:
    """Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 if it does not parse."""
    text = env_text(name)
    if text is None:
        return fallback
    try:
        parsed = parse(text)
    except (ValueError, TypeError):
        parsed = None
    if parsed is None:
        raise SystemExit(f'{name}="{text}" does not parse, expected {expected}')
    return parsed


def env_count(name: str, fallback: int) -> int:
    """Reads a positive count like `128`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, parse_count, "a positive count")


def env_duration(name: str, fallback: float) -> float:
    """Reads a duration like `200ms` or `10s`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s")


def env_flag(name: str, fallback: bool) -> bool:
    """Reads `0`, `1`, `true` or `false`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, {"0": False, "false": False, "1": True, "true": True}.get, "0, 1, true or false")


def env_seed(name: str, fallback: int) -> int:
    """Reads a 32-bit seed or `random`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random")


def parse_seed(text: str) -> int | None:
    """Parses a 32-bit unsigned integer, or `random` as 32 bits from the OS entropy source."""
    if text == "random":
        return secrets.randbits(32)
    return int(text) if re.fullmatch(r"[0-9]+", text) and int(text) < 1 << 32 else None


def parse_threads(text: str) -> int | None:
    """Parses a thread count like `8`, or `0` as every available core."""
    return (os.cpu_count() or 1) if text == "0" else parse_count(text)


def parse_count(text: str) -> int | None:
    """Parses a positive whole number in ASCII digits, like `128`; zero is `None`."""
    digits = re.fullmatch(r"[0-9]+", text) is not None
    return (int(text) or None) if digits else None


def parse_duration(text: str) -> float | None:
    """Parses a duration like `200ms` or `10s` into seconds; a bare number, a fraction or zero is `None`."""
    if text.endswith("ms"):
        count = parse_count(text.removesuffix("ms"))
        return None if count is None else count / 1000
    return parse_count(text.removesuffix("s")) if text.endswith("s") else None


def spell_duration(seconds: float) -> str:
    """Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`."""
    milliseconds = round(seconds * 1000)
    return f"{milliseconds // 1000}s" if milliseconds % 1000 == 0 else f"{milliseconds}ms"


# endregion: Environment variables

# Read at import, because BLAS-backed libraries size their pools when NumPy first loads them.
_THREADS = env_parsed("NUMWARS_THREADS", 1, parse_threads, "a count, 0 for all cores")
_POOLS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")
for _variable in _POOLS:
    os.environ[_variable] = str(_THREADS)

import numpy as np

try:
    import ml_dtypes

    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False


# region: Settings


@dataclass(frozen=True)
class Settings:
    """Every `NUMWARS_*` setting, read once at the top of `main` and passed down. Mirrors the Rust
    `Settings` field for field, less `sample_size` and plus `output`."""

    filter: str
    filter_pattern: re.Pattern[str] | None
    seed: int
    warmup: float
    time_limit: float
    batch_per_core: int
    threads: int
    dims: int
    dims_height: int
    dims_width: int
    dims_depth: int
    output: str

    def selects(self, name: str) -> bool:
        """Whether `NUMWARS_FILTER` selects the row `name`, as a regex or else as a substring."""
        return bool(self.filter_pattern.search(name)) if self.filter_pattern else self.filter in name

    @property
    def batch_size(self) -> int:
        """Items one call processes: the batch per core times the threads."""
        return self.batch_per_core * self.threads


def read_settings() -> Settings:
    """Reads every `NUMWARS_*` variable, exiting with status 1 on the first that does not parse."""
    filter = env_text("NUMWARS_FILTER") or ""
    try:
        filter_pattern = re.compile(filter) if filter else None
    except re.error:
        filter_pattern = None
    return Settings(
        filter=filter,
        filter_pattern=filter_pattern,
        seed=env_seed("NUMWARS_SEED", 42),
        warmup=env_duration("NUMWARS_WARMUP", 1.0),
        time_limit=env_duration("NUMWARS_TIME_LIMIT", 10.0),
        batch_per_core=env_count("NUMWARS_BATCH_PER_CORE", 2048),
        threads=_THREADS,
        dims=env_count("NUMWARS_DIMS", 1536),
        dims_height=env_count("NUMWARS_DIMS_HEIGHT", 1024),
        dims_width=env_count("NUMWARS_DIMS_WIDTH", 128),
        dims_depth=env_count("NUMWARS_DIMS_DEPTH", 1536),
        output=env_parsed("NUMWARS_OUTPUT", "table", {"table": "table", "json": "json"}.get, "table or json"),
    )


def print_settings(settings: Settings) -> None:
    """Prints every setting as "- Name: value", in the grammar it is read in."""
    print(f"- Seed: {settings.seed}")
    print(f"- Filter: {settings.filter or 'none'}")
    print(f"- Warm-up: {spell_duration(settings.warmup)}")
    print(f"- Time limit: {spell_duration(settings.time_limit)}")
    print(f"- Batch per core: {settings.batch_per_core}")
    print(f"- Threads: {settings.threads}")
    print(f"- Dims: {settings.dims}")
    print(f"- Dims height: {settings.dims_height}")
    print(f"- Dims width: {settings.dims_width}")
    print(f"- Dims depth: {settings.dims_depth}")
    print(f"- Output: {settings.output}")


# endregion: Settings

# region: Timing Utilities


def measure_average_duration(func: Callable[[], Any], settings: Settings) -> float:
    """Average seconds per call of `func`, after `settings.warmup` of warm-up, over `settings.time_limit`."""
    start = time.perf_counter()
    while (time.perf_counter() - start) < settings.warmup:
        func()

    durations = []
    start = time.perf_counter()
    while (time.perf_counter() - start) < settings.time_limit:
        call_start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - call_start)

    if not durations:
        call_start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - call_start)

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

# region: Data types


def get_ml_dtype(dtype_str: str):
    """Get ML data type from string (bf16, e4m3, e5m2, i4, u4)."""
    if not HAS_ML_DTYPES:
        raise ImportError("ml_dtypes package is required for exotic types. Install with: pip install ml_dtypes")

    dtype_map = {
        "bf16": ml_dtypes.bfloat16,
        "e4m3": ml_dtypes.float8_e4m3fn,
        "e5m2": ml_dtypes.float8_e5m2,
        "i4": ml_dtypes.int4,
        "u4": ml_dtypes.uint4,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown ML dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")

    return dtype_map[dtype_str]


def parse_numpy_dtype(dtype_str: str) -> np.dtype:
    """
    Parse a data type string to NumPy dtype.

    Supports: f64, f32, f16, bf16, e4m3, e5m2, i8, i4, u8, u4, complex64, complex128
    """
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

    if dtype_str in ("bf16", "e4m3", "e5m2", "i4", "u4"):
        return get_ml_dtype(dtype_str)

    raise ValueError(f"Unknown dtype: {dtype_str}")


# endregion

# region: Results Formatting


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


def print_results_table(results: list[dict], headers: list[str] | None = None) -> None:
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

        rows = [[r.get(h, "") for h in headers] for r in results]

        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        if not results:
            print("No results to display")
            return

        if headers is None:
            headers = list(results[0].keys())

        print(" | ".join(headers))
        print("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))

        for r in results:
            print(" | ".join(str(r.get(h, "")) for h in headers))


# endregion
