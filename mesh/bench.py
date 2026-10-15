#!/usr/bin/env python3
"""
Many-to-many shape distance benchmarks: NumKong vs Python alternatives.

Scores every row cloud against every column cloud with the correspondence-free distances that
replaced mesh alignment. Clouds pack once, outside the timed loop, centered and rotated onto their
principal axes, and every kind takes the best of the 8 sign flips of those axes:
- chamfers and hausdorffs: against SciPy's cKDTree over the same principal frames and flips
- wassersteinprocrustes: against SciPy's linear_sum_assignment alternated with orthogonal_procrustes,
  as between clouds of one size the exact transport plan is an assignment
- gromovwassersteins: against POT's gromov_wasserstein2 over the same distance matrices

Can be run with uv:
    uv run --with numkong,numpy,scipy,pot,tabulate mesh/bench.py

Or with traditional pip:
    pip install -e ".[mesh]"
    python mesh/bench.py
"""

import json
import sys
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from numwars import (
    calculate_mps,
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

try:
    import ot
except ImportError:
    ot = None

CLOUD_COUNT = 8
KINDS = ("chamfers", "hausdorffs", "wassersteinprocrustes", "gromovwassersteins")
SIGN_FLIPS = [np.array([-1.0 if flip >> axis & 1 else 1.0 for axis in range(3)]) for flip in range(8)]


@dataclass
class BenchmarkResult:
    workload: str
    library: str
    input_dtype: str
    output_dtype: str
    count: int
    points: int
    duration_secs: float
    primary_value: float


def build_clouds(points: int, dtype_str: str, rng: np.random.Generator) -> list[np.ndarray]:
    """`CLOUD_COUNT` standard-normal clouds of shape [points,3]."""
    return [rng.standard_normal((points, 3)).astype(parse_numpy_dtype(dtype_str)) for _ in range(CLOUD_COUNT)]


def principal_frame(cloud: np.ndarray) -> np.ndarray:
    """Centers `cloud` and rotates it onto its principal axes in descending spread, as NumKong packs it."""
    centered = cloud.astype(np.float64) - cloud.astype(np.float64).mean(axis=0)
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    return centered @ eigenvectors[:, ::-1]


def scipy_distance(workload: str, a: np.ndarray, b: np.ndarray, a_tree: cKDTree, b_tree: cKDTree) -> float:
    """Chamfer or Hausdorff distance between two principal frames, the best of the 8 sign flips."""
    best = np.inf
    for signs in SIGN_FLIPS:
        # A flip is its own inverse, so each direction flips the points it queries
        forward, reverse = b_tree.query(a * signs)[0], a_tree.query(b * signs)[0]
        if workload == "chamfers":
            best = min(best, (forward.mean() + reverse.mean()) / 2)
        else:
            best = min(best, max(forward.max(), reverse.max()))
    return best


def scipy_wasserstein_procrustes(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-Procrustes between two principal frames of one size: from each of the 8 sign flips,
    the optimal assignment alternates with the Procrustes fit of its pairs while the cost falls."""
    best = np.inf
    for signs in SIGN_FLIPS:
        rotation, cost = np.diag(signs), np.inf
        while True:
            squares = cdist(a, b @ rotation, "sqeuclidean")
            rows, columns = linear_sum_assignment(squares)
            matched = squares[rows, columns].mean()
            if not matched < cost:
                break
            cost = matched
            rotation = orthogonal_procrustes(b[columns], a[rows])[0]
            # Like NumKong, each start stays among the rotations, or among the reflections, it began with
            if np.linalg.det(rotation) * signs.prod() < 0:
                left, _, right = np.linalg.svd(b[columns].T @ a[rows])
                rotation = left @ np.diag([1.0, 1.0, -1.0]) @ right
        best = min(best, cost)
    return np.sqrt(best)


def pairwise(rows: list, columns: list, distance) -> np.ndarray:
    """The [rows,columns] matrix of `distance` from every row to every column."""
    return np.array([[distance(row, column) for column in columns] for row in rows])


def build_candidates(dtype_str: str, points: int, threads: int, rng: np.random.Generator) -> list[tuple]:
    """Every (workload, library, output dtype, callable) row over one tile of fresh clouds."""
    rows, columns = build_clouds(points, dtype_str, rng), build_clouds(points, dtype_str, rng)
    row_pack, column_pack = nk.shapes_pack(rows, threads=threads), nk.shapes_pack(columns, threads=threads)
    output_dtype = {"f32": "f64", "f16": "f32"}[dtype_str]
    candidates = [
        (kind, "NumKong", output_dtype, partial(getattr(nk, f"{kind}_packed"), row_pack, column_pack, threads=threads))
        for kind in KINDS
    ]

    row_frames, column_frames = (
        [principal_frame(cloud) for cloud in rows],
        [principal_frame(cloud) for cloud in columns],
    )
    row_trees, column_trees = [cKDTree(frame) for frame in row_frames], [cKDTree(frame) for frame in column_frames]
    for kind in ("chamfers", "hausdorffs"):
        candidates.append(
            (
                kind,
                "SciPy",
                "f64",
                lambda kind=kind: pairwise(
                    list(zip(row_frames, row_trees)),
                    list(zip(column_frames, column_trees)),
                    lambda row, column: scipy_distance(kind, row[0], column[0], row[1], column[1]),
                ),
            )
        )
    # f16 packs round their points to f16, which can tip a pair into another local optimum
    if dtype_str == "f32":
        expected = np.asarray(nk.wassersteinprocrustes_packed(row_pack, column_pack))[0, 0]
        actual = scipy_wasserstein_procrustes(row_frames[0], column_frames[0])
        assert np.isclose(actual, expected, rtol=1e-6), f"SciPy {actual} vs NumKong {expected}"
    candidates.append(
        (
            "wassersteinprocrustes",
            "SciPy",
            "f64",
            lambda: pairwise(row_frames, column_frames, scipy_wasserstein_procrustes),
        )
    )
    if ot is not None:
        row_distances = [cdist(frame, frame) for frame in row_frames]
        column_distances = [cdist(frame, frame) for frame in column_frames]
        weights = np.full(points, 1 / points)
        candidates.append(
            (
                "gromovwassersteins",
                "POT",
                "f64",
                lambda: pairwise(
                    row_distances,
                    column_distances,
                    lambda row, column: ot.gromov.gromov_wasserstein2(row, column, weights, weights, "square_loss"),
                ),
            )
        )
    return candidates


def benchmark_id(workload: str, library: str, input_dtype: str, points: int) -> str:
    return f"mesh/{workload}/{library.lower()}/{input_dtype}/{CLOUD_COUNT}x{CLOUD_COUNT}x{points}"


def result_to_entry(result: BenchmarkResult) -> dict:
    return {
        "suite": "mesh",
        "workload": result.workload,
        "benchmark_id": benchmark_id(result.workload, result.library, result.input_dtype, result.points),
        "library": result.library,
        "dtype": result.input_dtype,
        "input_dtype": result.input_dtype,
        "output_dtype": result.output_dtype,
        "display_signature": f"{result.input_dtype} \u2192 {result.output_dtype}",
        "count": result.count,
        "points": result.points,
        "primary_value": result.primary_value,
        "unit": "MP/s",
        "duration_secs": result.duration_secs,
    }


def main() -> None:
    settings = read_settings("mesh")
    if settings.output == "table":
        print_settings(settings)
    points = settings.shape_points
    count = CLOUD_COUNT * CLOUD_COUNT

    rng = np.random.default_rng(settings.seed)
    results = []
    for input_dtype in ("f32", "f16"):
        for workload, library, output_dtype, func in build_candidates(input_dtype, points, settings.threads, rng):
            if not settings.selects(benchmark_id(workload, library, input_dtype, points)):
                continue
            duration = measure_average_duration(func, settings)
            result = BenchmarkResult(
                workload=workload,
                library=library,
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                count=count,
                points=points,
                duration_secs=duration,
                primary_value=calculate_mps(count, duration),
            )
            results.append(result_to_entry(result))

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
