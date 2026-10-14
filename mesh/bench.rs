//! Benchmark for many-to-many shape distances between 3D point clouds
//!
//! Scores every row cloud against every column cloud with the correspondence-free distances that
//! replaced mesh alignment: Chamfer, Hausdorff, Wasserstein-Procrustes and Gromov-Wasserstein.
//! Clouds pack once, outside the timed loop, centered and rotated onto their principal axes, and
//! every kind takes the best of the 8 sign flips of those axes. The nalgebra baseline packs the
//! same way with `SymmetricEigen` and searches the same flips by brute force. The two transport
//! kinds have no Rust competitor.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_mesh --bench bench_mesh
//! NUMWARS_FILTER="mesh/chamfers" NUMWARS_SHAPE_POINTS=128 cargo bench --features bench_mesh
//! ```
//!
//! Benchmark naming: mesh/{kind}/{dtype}/{rows}x{columns}x{points}
//! Examples: mesh/chamfers/f32/8x8x32, mesh/gromovwassersteins/f16/8x8x32

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use numkong::{f16, Capabilities, NumberLike, Shapes, ShapesPackedMatrix};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use numwars::Settings;

/// Row and column clouds per call, the square tile NumKong's own benchmarks time.
const CLOUD_COUNT: usize = 8;

// region: Operation Model

#[derive(Clone, Copy)]
enum ShapeKind {
    Chamfers,
    Hausdorffs,
    WassersteinProcrustes,
    GromovWassersteins,
}

impl ShapeKind {
    fn slug(self) -> &'static str {
        match self {
            ShapeKind::Chamfers => "chamfers",
            ShapeKind::Hausdorffs => "hausdorffs",
            ShapeKind::WassersteinProcrustes => "wassersteinprocrustes",
            ShapeKind::GromovWassersteins => "gromovwassersteins",
        }
    }
}

// endregion

// region: Baseline Implementations

/// Centers `cloud` on its centroid and rotates it onto its principal axes in descending spread,
/// as NumKong packs it.
fn baseline_pack<T: NumberLike + Copy>(cloud: &[[T; 3]]) -> Vec<Vector3<f64>> {
    let points: Vec<Vector3<f64>> = cloud
        .iter()
        .map(|point| Vector3::from_fn(|axis, _| point[axis].to_f64()))
        .collect();
    let count = points.len() as f64;
    let centroid = points.iter().sum::<Vector3<f64>>() / count;
    let covariance = points
        .iter()
        .map(|point| (point - centroid) * (point - centroid).transpose())
        .sum::<Matrix3<f64>>()
        / count;
    let eigen = SymmetricEigen::new(covariance);
    let mut order = [0, 1, 2];
    order.sort_by(|&first, &second| eigen.eigenvalues[second].total_cmp(&eigen.eigenvalues[first]));
    let axes = Matrix3::from_columns(&order.map(|axis| eigen.eigenvectors.column(axis).into_owned()));
    points
        .iter()
        .map(|point| axes.transpose() * (point - centroid))
        .collect()
}

/// The mean and the largest distance from every point of `from` to its nearest point of `to`
/// flipped by `signs`.
fn baseline_nearest(from: &[Vector3<f64>], to: &[Vector3<f64>], signs: &Vector3<f64>) -> (f64, f64) {
    let (mut sum, mut largest) = (0.0, 0.0f64);
    for point in from {
        let squares = to
            .iter()
            .map(|other| (point - other.component_mul(signs)).norm_squared());
        let nearest = squares.fold(f64::INFINITY, f64::min).sqrt();
        sum += nearest;
        largest = largest.max(nearest);
    }
    (sum / from.len() as f64, largest)
}

/// Chamfer or Hausdorff distance between two packed clouds, the best of the 8 sign flips.
fn baseline_distance(kind: ShapeKind, a: &[Vector3<f64>], b: &[Vector3<f64>]) -> f64 {
    let flipped = |flip: usize| {
        let signs = Vector3::from_fn(|axis, _| if flip >> axis & 1 == 1 { -1.0 } else { 1.0 });
        // A flip is its own inverse, so the reverse direction flips `a` instead of `b`
        let (forward_mean, forward_largest) = baseline_nearest(a, b, &signs);
        let (reverse_mean, reverse_largest) = baseline_nearest(b, a, &signs);
        match kind {
            ShapeKind::Chamfers => (forward_mean + reverse_mean) / 2.0,
            _ => forward_largest.max(reverse_largest),
        }
    };
    (0..8).map(flipped).fold(f64::INFINITY, f64::min)
}

// endregion

// region: Data Generation

/// Standard-normal points through Box-Muller, like `standard_normal` in `bench.py`.
fn generate_point_cloud<T: NumberLike>(rng: &mut impl Rng, count: usize) -> Vec<[T; 3]> {
    let mut normal = || {
        let (radius, angle) = (1.0 - rng.random::<f64>(), rng.random::<f64>());
        T::from_f64((-2.0 * radius.ln()).sqrt() * (std::f64::consts::TAU * angle).cos())
    };
    (0..count).map(|_| [normal(), normal(), normal()]).collect()
}

// endregion

// region: Per-Library Runs

fn run_numkong<T: Shapes>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    kind: ShapeKind,
    rows: &[Vec<[T; 3]>],
    columns: &[Vec<[T; 3]>],
) {
    let row_slices: Vec<&[[T; 3]]> = rows.iter().map(Vec::as_slice).collect();
    let column_slices: Vec<&[[T; 3]]> = columns.iter().map(Vec::as_slice).collect();
    let row_pack = ShapesPackedMatrix::<T>::new(&row_slices).expect("Failed to pack rows");
    let column_pack = ShapesPackedMatrix::<T>::new(&column_slices).expect("Failed to pack columns");
    let mut distances = vec![T::Distance::default(); rows.len() * columns.len()];
    group.bench_function("numkong", |bench| {
        bench.iter(|| {
            match kind {
                ShapeKind::Chamfers => row_pack.chamfers(&column_pack, &mut distances, None),
                ShapeKind::Hausdorffs => row_pack.hausdorffs(&column_pack, &mut distances, None),
                ShapeKind::WassersteinProcrustes => row_pack.wassersteinprocrustes(&column_pack, &mut distances, None),
                ShapeKind::GromovWassersteins => row_pack.gromovwassersteins(&column_pack, &mut distances, None),
            }
            .expect("Shape distances failed");
            black_box(&distances);
        })
    });
}

fn run_nalgebra<T: NumberLike + Copy>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    kind: ShapeKind,
    rows: &[Vec<[T; 3]>],
    columns: &[Vec<[T; 3]>],
) {
    let row_packs: Vec<_> = rows.iter().map(|cloud| baseline_pack(cloud)).collect();
    let column_packs: Vec<_> = columns.iter().map(|cloud| baseline_pack(cloud)).collect();
    let mut distances = vec![0.0f64; rows.len() * columns.len()];
    group.bench_function("nalgebra", |bench| {
        bench.iter(|| {
            for (row_distances, row) in distances.chunks_exact_mut(column_packs.len()).zip(&row_packs) {
                for (distance, column) in row_distances.iter_mut().zip(&column_packs) {
                    *distance = baseline_distance(kind, row, column);
                }
            }
            black_box(&distances);
        })
    });
}

// endregion

// region: Benchmarks

fn bench_kind_dtype<T: Shapes + NumberLike + Copy>(
    c: &mut Criterion,
    settings: &Settings,
    kind: ShapeKind,
    dtype: &str,
) {
    let points = settings.shape_points;
    let name = format!("mesh/{}/{dtype}/{CLOUD_COUNT}x{CLOUD_COUNT}x{points}", kind.slug());
    if !settings.selects(&name) {
        return;
    }
    let mut rng = StdRng::seed_from_u64(settings.seed.into());
    let mut clouds = || {
        let clouds = (0..CLOUD_COUNT).map(|_| generate_point_cloud::<T>(&mut rng, points));
        clouds.collect::<Vec<_>>()
    };
    let (rows, columns) = (clouds(), clouds());

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements((CLOUD_COUNT * CLOUD_COUNT) as u64));
    run_numkong(&mut group, kind, &rows, &columns);
    if let ShapeKind::Chamfers | ShapeKind::Hausdorffs = kind {
        run_nalgebra(&mut group, kind, &rows, &columns);
    }
    group.finish();
}

/// Benchmark every shape kind over `f32` clouds, scored in `f64`, and `f16` clouds, scored in `f32`.
pub fn bench_shapes(c: &mut Criterion, settings: &Settings) {
    let kinds = [
        ShapeKind::Chamfers,
        ShapeKind::Hausdorffs,
        ShapeKind::WassersteinProcrustes,
        ShapeKind::GromovWassersteins,
    ];
    for kind in kinds {
        bench_kind_dtype::<f32>(c, settings, kind, "f32");
        bench_kind_dtype::<f16>(c, settings, kind, "f16");
    }
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn baseline_ignores_rotation_and_translation() {
        let cloud = [
            [3.0f32, 0.5, 0.0],
            [-3.0, 0.0, 0.2],
            [0.0, 2.0, 0.0],
            [0.5, -2.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let moved = cloud.map(|[x, y, z]| [10.0 - y, x, z + 5.0]);
        let (a, b) = (baseline_pack(&cloud), baseline_pack(&moved));
        assert!(baseline_distance(ShapeKind::Chamfers, &a, &b) < 1e-5);
        assert!(baseline_distance(ShapeKind::Hausdorffs, &a, &b) < 1e-5);
    }

    #[test]
    fn baseline_hausdorff_bounds_chamfer() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = baseline_pack(&generate_point_cloud::<f32>(&mut rng, 32));
        let b = baseline_pack(&generate_point_cloud::<f32>(&mut rng, 24));
        let chamfer = baseline_distance(ShapeKind::Chamfers, &a, &b);
        assert!(chamfer > 0.0 && chamfer <= baseline_distance(ShapeKind::Hausdorffs, &a, &b));
    }
}

// endregion

// region: Main

fn main() {
    let settings = Settings::read();
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    let mut criterion = numwars::configure_criterion(&settings);
    bench_shapes(&mut criterion, &settings);
    criterion.final_summary();
}

// endregion
