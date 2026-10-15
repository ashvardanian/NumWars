#!/usr/bin/env python3
"""
Many-to-many geospatial distance benchmarks: NumKong vs Python alternatives.

Scores every row point against every column point into a [rows,columns] matrix:
- haversines: great-circle metres on the IUGG mean sphere, against scikit-learn and NumPy
- vincentys: geodesic metres on the WGS-84 ellipsoid, against pyproj's Karney solver
- sqeuclideans3d: squared distances between Earth-fixed points, against SciPy's cdist

Each point set packs once, outside the timed loop, as the dots suite packs its B operand.

Can be run with uv:
    uv run --with numkong,numpy,scipy,scikit-learn,pyproj,tabulate geospatial/bench.py

Or with traditional pip:
    pip install -e ".[geospatial]"
    python geospatial/bench.py
"""

import json
import sys
from dataclasses import dataclass

import numpy as np
from pyproj import Geod
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances

from numwars import (
    Settings,
    calculate_mps,
    measure_average_duration,
    normalize_dtype_name,
    print_results_table,
    print_settings,
    read_settings,
)

try:
    import numkong as nk
except ImportError:
    print("Error: numkong not found. Install with: pip install numkong")
    sys.exit(1)


EARTH_RADIUS_METERS = 6_371_008.8
EARTH_RADIUS_CENTIMETERS = 637_100_000


@dataclass
class BenchmarkResult:
    workload: str
    library: str
    input_dtype: str
    output_dtype: str
    rows: int
    columns: int
    duration_secs: float
    primary_value: float


def build_coords(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes in radians, as NumKong and scikit-learn take them."""
    latitudes = np.radians(rng.uniform(-90.0, 90.0, size=count))
    longitudes = np.radians(rng.uniform(-180.0, 180.0, size=count))
    return latitudes, longitudes


def build_points(count: int, dtype_str: str, rng: np.random.Generator) -> np.ndarray:
    """Earth-fixed points: `i32` centimetres or `f64` metres within the Earth's radius on every axis."""
    if dtype_str == "i32":
        return rng.integers(-EARTH_RADIUS_CENTIMETERS, EARTH_RADIUS_CENTIMETERS, size=(count, 3), dtype=np.int32)
    return rng.uniform(-EARTH_RADIUS_CENTIMETERS / 100, EARTH_RADIUS_CENTIMETERS / 100, size=(count, 3))


def numpy_haversines(rows: tuple[np.ndarray, np.ndarray], columns: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Great-circle metres between every row and column point, broadcast over a [rows,columns] grid."""
    row_latitudes, row_longitudes = rows[0][:, None], rows[1][:, None]
    column_latitudes, column_longitudes = columns[0][None, :], columns[1][None, :]
    half_chord_squared = (
        np.sin((column_latitudes - row_latitudes) / 2) ** 2
        + np.cos(row_latitudes) * np.cos(column_latitudes) * np.sin((column_longitudes - row_longitudes) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(half_chord_squared))


def benchmark(
    workload: str, library: str, input_dtype: str, rows: int, columns: int, func, settings: Settings
) -> BenchmarkResult:
    output_dtype = normalize_dtype_name(np.asarray(func()).dtype)
    duration = measure_average_duration(func, settings)
    return BenchmarkResult(
        workload=workload,
        library=library,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        rows=rows,
        columns=columns,
        duration_secs=duration,
        primary_value=calculate_mps(rows * columns, duration),
    )


def benchmark_id(workload: str, library: str, input_dtype: str, rows: int, columns: int) -> str:
    return f"geospatial/{workload}/{library.lower()}/{input_dtype}/{rows}x{columns}"


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "geospatial",
        "workload": result.workload,
        "benchmark_id": benchmark_id(result.workload, result.library, result.input_dtype, result.rows, result.columns),
        "library": result.library,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_signature": f"{result.input_dtype} \u2192 {result.output_dtype}",
        "height": result.rows,
        "width": result.columns,
        "primary_value": result.primary_value,
        "unit": "MP/s",
        "duration_secs": result.duration_secs,
    }


def main() -> None:
    settings = read_settings("geospatial")
    if settings.output == "table":
        print_settings(settings)
    rows, columns = settings.dims_height, settings.dims_width

    rng = np.random.default_rng(settings.seed)
    row_coords, column_coords = build_coords(rows, rng), build_coords(columns, rng)
    haversine_rows, haversine_columns = nk.haversines_pack(*row_coords), nk.haversines_pack(*column_coords)
    vincenty_rows, vincenty_columns = nk.vincentys_pack(*row_coords), nk.vincentys_pack(*column_coords)
    sklearn_rows, sklearn_columns = np.column_stack(row_coords), np.column_stack(column_coords)
    # pyproj pairs its inputs elementwise, so the grid is spelled out once, in degrees
    row_latitudes, row_longitudes = (np.repeat(np.degrees(axis), columns) for axis in row_coords)
    column_latitudes, column_longitudes = (np.tile(np.degrees(axis), rows) for axis in column_coords)
    geodesic = Geod(ellps="WGS84")
    points = {dtype: (build_points(rows, dtype, rng), build_points(columns, dtype, rng)) for dtype in ("i32", "f64")}

    benchmarks = [
        ("haversines", "NumKong", "f64", lambda: nk.haversines_packed(haversine_rows, haversine_columns)),
        (
            "haversines",
            "scikit-learn",
            "f64",
            lambda: haversine_distances(sklearn_rows, sklearn_columns) * EARTH_RADIUS_METERS,
        ),
        ("haversines", "NumPy", "f64", lambda: numpy_haversines(row_coords, column_coords)),
        ("vincentys", "NumKong", "f64", lambda: nk.vincentys_packed(vincenty_rows, vincenty_columns)),
        (
            "vincentys",
            "pyproj",
            "f64",
            lambda: geodesic.inv(row_longitudes, row_latitudes, column_longitudes, column_latitudes)[2],
        ),
        ("sqeuclideans3d", "NumKong", "i32", lambda: nk.sqeuclideans3d(*points["i32"], threads=settings.threads)),
        ("sqeuclideans3d", "SciPy", "i32", lambda: cdist(*points["i32"], "sqeuclidean")),
        ("sqeuclideans3d", "NumKong", "f64", lambda: nk.sqeuclideans3d(*points["f64"], threads=settings.threads)),
        ("sqeuclideans3d", "SciPy", "f64", lambda: cdist(*points["f64"], "sqeuclidean")),
    ]

    results = []
    for workload, library, input_dtype, func in benchmarks:
        if not settings.selects(benchmark_id(workload, library, input_dtype, rows, columns)):
            continue
        results.append(result_to_entry(benchmark(workload, library, input_dtype, rows, columns, func, settings)))

    if settings.output == "json":
        json.dump({"results": results}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if not results:
        print("No benchmarks matched the filter.")
        return

    table_rows = [
        {
            "benchmark": entry["benchmark_id"],
            "library": entry["library"],
            "signature": entry["display_signature"],
            "duration": f"{entry['duration_secs'] * 1e3:.3f} ms",
            "throughput": f"{entry['primary_value']:.2f} {entry['unit']}",
        }
        for entry in results
    ]
    print_results_table(table_rows)


if __name__ == "__main__":
    main()
