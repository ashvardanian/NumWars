//! Benchmark for mesh alignment operations (RMSD, Kabsch, Umeyama)
//!
//! Compares NumKong vs nalgebra-based baseline implementations for 3D point cloud
//! alignment algorithms. Mini-float types (`f16`, `bf16`) are numkong-first.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_mesh --bench bench_mesh
//! NUMWARS_FILTER="mesh/rmsd" cargo bench --features bench_mesh
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Number of 3D points per cloud (default: 2048)
//! - NUMWARS_FILTER: Regex to filter benchmark names
//!
//! Benchmark naming: mesh/{operation}/{dtype}
//! Examples: mesh/rmsd/f32, mesh/kabsch/f64, mesh/umeyama/bf16

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use nalgebra::Matrix3;
use num_traits::Float;
use numkong::{bf16, capabilities, f16, MeshAlignment};
use rand::Rng;
use std::hint::black_box;
use utils::*;

// region: Operation Model

#[derive(Clone, Copy)]
enum MeshOp {
    Rmsd,
    Kabsch,
    Umeyama,
}

impl MeshOp {
    fn slug(self) -> &'static str {
        match self {
            MeshOp::Rmsd => "rmsd",
            MeshOp::Kabsch => "kabsch",
            MeshOp::Umeyama => "umeyama",
        }
    }
}

// endregion

// region: Baseline Implementations

/// Baseline RMSD: simple point-to-point root mean square deviation (no alignment).
fn baseline_rmsd<T: Float>(a_points: &[[T; 3]], b_points: &[[T; 3]]) -> T {
    let n = T::from(a_points.len()).unwrap();
    let mut sum_sq = T::zero();
    for (a, b) in a_points.iter().zip(b_points) {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        sum_sq = sum_sq + dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / n).sqrt()
}

/// RMSD baseline variant for types accumulated through f32 conversion.
fn baseline_rmsd_f32<T: BaselineConvert<f32> + Copy>(a_points: &[[T; 3]], b_points: &[[T; 3]]) -> f32 {
    let n = a_points.len() as f32;
    let mut sum_sq = 0.0f32;
    for (a, b) in a_points.iter().zip(b_points) {
        let dx = a[0].to_acc() - b[0].to_acc();
        let dy = a[1].to_acc() - b[1].to_acc();
        let dz = a[2].to_acc() - b[2].to_acc();
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / n).sqrt()
}

/// Baseline Kabsch algorithm: optimal rotation via SVD to minimize RMSD.
///
/// Returns (rotation_matrix, rmsd) where rotation_matrix is [f64; 9] in row-major order.
fn baseline_kabsch<T: Float + Into<f64> + 'static>(a_points: &[[T; 3]], b_points: &[[T; 3]]) -> ([f64; 9], f64)
where
    f64: From<T>,
{
    let n = a_points.len() as f64;

    let (mut ca, mut cb) = ([0.0f64; 3], [0.0f64; 3]);
    for (a, b) in a_points.iter().zip(b_points) {
        for k in 0..3 {
            ca[k] += f64::from(a[k]);
            cb[k] += f64::from(b[k]);
        }
    }
    for k in 0..3 {
        ca[k] /= n;
        cb[k] /= n;
    }

    let mut h = Matrix3::<f64>::zeros();
    for (a, b) in a_points.iter().zip(b_points) {
        let ac = [
            f64::from(a[0]) - ca[0],
            f64::from(a[1]) - ca[1],
            f64::from(a[2]) - ca[2],
        ];
        let bc = [
            f64::from(b[0]) - cb[0],
            f64::from(b[1]) - cb[1],
            f64::from(b[2]) - cb[2],
        ];
        for row in 0..3 {
            for col in 0..3 {
                h[(row, col)] += ac[row] * bc[col];
            }
        }
    }

    let svd = h.svd(true, true);
    let u = svd.u.expect("SVD U missing");
    let v_t = svd.v_t.expect("SVD Vt missing");

    let v = v_t.transpose();
    let d = (v * u.transpose()).determinant();
    let sign = if d < 0.0 { -1.0 } else { 1.0 };
    let diag = Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, sign));
    let rotation = v * diag * u.transpose();

    let mut sum_sq = 0.0f64;
    for (a, b) in a_points.iter().zip(b_points) {
        let ac = [
            f64::from(a[0]) - ca[0],
            f64::from(a[1]) - ca[1],
            f64::from(a[2]) - ca[2],
        ];
        let bc = [
            f64::from(b[0]) - cb[0],
            f64::from(b[1]) - cb[1],
            f64::from(b[2]) - cb[2],
        ];
        let ra = [
            rotation[(0, 0)] * ac[0] + rotation[(0, 1)] * ac[1] + rotation[(0, 2)] * ac[2],
            rotation[(1, 0)] * ac[0] + rotation[(1, 1)] * ac[1] + rotation[(1, 2)] * ac[2],
            rotation[(2, 0)] * ac[0] + rotation[(2, 1)] * ac[1] + rotation[(2, 2)] * ac[2],
        ];
        let dx = ra[0] - bc[0];
        let dy = ra[1] - bc[1];
        let dz = ra[2] - bc[2];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    let rmsd = (sum_sq / n).sqrt();

    let mut rot_arr = [0.0f64; 9];
    for row in 0..3 {
        for col in 0..3 {
            rot_arr[row * 3 + col] = rotation[(row, col)];
        }
    }

    (rot_arr, rmsd)
}

/// Baseline Umeyama algorithm: Kabsch with uniform scale estimation.
///
/// Returns (rotation_matrix, scale, rmsd).
fn baseline_umeyama<T: Float + Into<f64> + 'static>(a_points: &[[T; 3]], b_points: &[[T; 3]]) -> ([f64; 9], f64, f64)
where
    f64: From<T>,
{
    let n = a_points.len() as f64;

    let (mut ca, mut cb) = ([0.0f64; 3], [0.0f64; 3]);
    for (a, b) in a_points.iter().zip(b_points) {
        for k in 0..3 {
            ca[k] += f64::from(a[k]);
            cb[k] += f64::from(b[k]);
        }
    }
    for k in 0..3 {
        ca[k] /= n;
        cb[k] /= n;
    }

    let mut h = Matrix3::<f64>::zeros();
    let mut var_a = 0.0f64;
    for (a, b) in a_points.iter().zip(b_points) {
        let ac = [
            f64::from(a[0]) - ca[0],
            f64::from(a[1]) - ca[1],
            f64::from(a[2]) - ca[2],
        ];
        let bc = [
            f64::from(b[0]) - cb[0],
            f64::from(b[1]) - cb[1],
            f64::from(b[2]) - cb[2],
        ];
        var_a += ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
        for row in 0..3 {
            for col in 0..3 {
                h[(row, col)] += ac[row] * bc[col];
            }
        }
    }
    var_a /= n;

    let svd = h.svd(true, true);
    let u = svd.u.expect("SVD U missing");
    let v_t = svd.v_t.expect("SVD Vt missing");
    let singular_values = svd.singular_values;

    let v = v_t.transpose();
    let d = (v * u.transpose()).determinant();
    let sign = if d < 0.0 { -1.0 } else { 1.0 };
    let diag = Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, sign));
    let rotation = v * diag * u.transpose();

    let trace_rh = singular_values[0] + singular_values[1] + sign * singular_values[2];
    let scale = trace_rh / (n * var_a);

    let mut sum_sq = 0.0f64;
    for (a, b) in a_points.iter().zip(b_points) {
        let ac = [
            f64::from(a[0]) - ca[0],
            f64::from(a[1]) - ca[1],
            f64::from(a[2]) - ca[2],
        ];
        let bc = [
            f64::from(b[0]) - cb[0],
            f64::from(b[1]) - cb[1],
            f64::from(b[2]) - cb[2],
        ];
        let ra = [
            scale * (rotation[(0, 0)] * ac[0] + rotation[(0, 1)] * ac[1] + rotation[(0, 2)] * ac[2]),
            scale * (rotation[(1, 0)] * ac[0] + rotation[(1, 1)] * ac[1] + rotation[(1, 2)] * ac[2]),
            scale * (rotation[(2, 0)] * ac[0] + rotation[(2, 1)] * ac[1] + rotation[(2, 2)] * ac[2]),
        ];
        let dx = ra[0] - bc[0];
        let dy = ra[1] - bc[1];
        let dz = ra[2] - bc[2];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    let rmsd = (sum_sq / n).sqrt();

    let mut rot_arr = [0.0f64; 9];
    for row in 0..3 {
        for col in 0..3 {
            rot_arr[row * 3 + col] = rotation[(row, col)];
        }
    }

    (rot_arr, scale, rmsd)
}

// endregion

// region: Data Generation

fn generate_point_cloud<T>(rng: &mut impl Rng, count: usize) -> Vec<[T; 3]> {
    generate_random::<[T; 3]>(rng, count)
}

fn get_point_count() -> usize {
    get_env_parsed("NUMWARS_MESH_POINTS", get_vector_dims())
}

// endregion

// region: Per-Library Run Traits

trait RunBaseline: Sized {
    fn run(_op: MeshOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[[Self; 3]], _b: &[[Self; 3]]) {}
}

impl RunBaseline for f32 {
    fn run(op: MeshOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[[f32; 3]], b: &[[f32; 3]]) {
        match op {
            MeshOp::Rmsd => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_rmsd(a, b))));
            }
            MeshOp::Kabsch => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_kabsch(a, b))));
            }
            MeshOp::Umeyama => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_umeyama(a, b))));
            }
        }
    }
}

impl RunBaseline for f64 {
    fn run(op: MeshOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[[f64; 3]], b: &[[f64; 3]]) {
        match op {
            MeshOp::Rmsd => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_rmsd(a, b))));
            }
            MeshOp::Kabsch => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_kabsch(a, b))));
            }
            MeshOp::Umeyama => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_umeyama(a, b))));
            }
        }
    }
}

impl RunBaseline for f16 {
    fn run(op: MeshOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[[f16; 3]], b: &[[f16; 3]]) {
        if let MeshOp::Rmsd = op {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_rmsd_f32(a, b))));
        }
    }
}

impl RunBaseline for bf16 {
    fn run(op: MeshOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[[bf16; 3]], b: &[[bf16; 3]]) {
        if let MeshOp::Rmsd = op {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_rmsd_f32(a, b))));
        }
    }
}

trait RunNalgebra: Sized {
    fn run(_op: MeshOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[[Self; 3]], _b: &[[Self; 3]]) {}
}

impl RunNalgebra for f32 {
    fn run(op: MeshOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[[f32; 3]], b: &[[f32; 3]]) {
        match op {
            MeshOp::Rmsd => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(baseline_rmsd(a, b))));
            }
            MeshOp::Kabsch => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(baseline_kabsch(a, b))));
            }
            MeshOp::Umeyama => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(baseline_umeyama(a, b))));
            }
        }
    }
}

impl RunNalgebra for f64 {
    fn run(op: MeshOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[[f64; 3]], b: &[[f64; 3]]) {
        match op {
            MeshOp::Rmsd => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(baseline_rmsd(a, b))));
            }
            MeshOp::Kabsch => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(baseline_kabsch(a, b))));
            }
            MeshOp::Umeyama => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(baseline_umeyama(a, b))));
            }
        }
    }
}

impl RunNalgebra for f16 {}
impl RunNalgebra for bf16 {}

trait RunNumKong: MeshAlignment + Sized {
    fn run(op: MeshOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[[Self; 3]], b: &[[Self; 3]]) {
        match op {
            MeshOp::Rmsd => {
                group.bench_function("numkong", |bench| bench.iter(|| black_box(MeshAlignment::rmsd(a, b))));
            }
            MeshOp::Kabsch => {
                group.bench_function("numkong", |bench| bench.iter(|| black_box(MeshAlignment::kabsch(a, b))));
            }
            MeshOp::Umeyama => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(MeshAlignment::umeyama(a, b)))
                });
            }
        }
    }
}

impl RunNumKong for f32 {}
impl RunNumKong for f64 {}
impl RunNumKong for f16 {}
impl RunNumKong for bf16 {}

// endregion

// region: Generic Helpers

fn bench_mesh_op_dtype<T>(c: &mut Criterion, rng: &mut impl Rng, op: MeshOp, dtype: &str, count: usize)
where
    T: RunBaseline + RunNalgebra + RunNumKong + Clone + 'static,
{
    let name = format!("mesh/{}/{}", op.slug(), dtype);
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((2 * count * std::mem::size_of::<[T; 3]>()) as u64));

    let a = generate_point_cloud::<T>(rng, count);
    let b = generate_point_cloud::<T>(rng, count);

    <T as RunNumKong>::run(op, &mut group, &a, &b);
    <T as RunNalgebra>::run(op, &mut group, &a, &b);
    <T as RunBaseline>::run(op, &mut group, &a, &b);

    group.finish();
}

// endregion

// region: Benchmarks

/// Benchmark RMSD (root mean square deviation without alignment).
pub fn bench_rmsd(c: &mut Criterion) {
    capabilities::configure_thread();
    let count = get_point_count();
    let mut rng = rand::rng();
    bench_mesh_op_dtype::<f32>(c, &mut rng, MeshOp::Rmsd, "f32", count);
    bench_mesh_op_dtype::<f64>(c, &mut rng, MeshOp::Rmsd, "f64", count);
    bench_mesh_op_dtype::<f16>(c, &mut rng, MeshOp::Rmsd, "f16", count);
    bench_mesh_op_dtype::<bf16>(c, &mut rng, MeshOp::Rmsd, "bf16", count);
}

/// Benchmark Kabsch algorithm (optimal rotation alignment).
pub fn bench_kabsch(c: &mut Criterion) {
    let count = get_point_count();
    let mut rng = rand::rng();
    bench_mesh_op_dtype::<f32>(c, &mut rng, MeshOp::Kabsch, "f32", count);
    bench_mesh_op_dtype::<f64>(c, &mut rng, MeshOp::Kabsch, "f64", count);
    bench_mesh_op_dtype::<f16>(c, &mut rng, MeshOp::Kabsch, "f16", count);
    bench_mesh_op_dtype::<bf16>(c, &mut rng, MeshOp::Kabsch, "bf16", count);
}

/// Benchmark Umeyama algorithm (Kabsch + uniform scale).
pub fn bench_umeyama(c: &mut Criterion) {
    let count = get_point_count();
    let mut rng = rand::rng();
    bench_mesh_op_dtype::<f32>(c, &mut rng, MeshOp::Umeyama, "f32", count);
    bench_mesh_op_dtype::<f64>(c, &mut rng, MeshOp::Umeyama, "f64", count);
    bench_mesh_op_dtype::<f16>(c, &mut rng, MeshOp::Umeyama, "f16", count);
    bench_mesh_op_dtype::<bf16>(c, &mut rng, MeshOp::Umeyama, "bf16", count);
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn rmsd_baseline_identical_points() {
        let points = vec![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let rmsd = baseline_rmsd(&points, &points);
        assert!(rmsd.abs() < 1e-12);
    }

    #[test]
    fn rmsd_baseline_known_value() {
        let a = vec![[0.0f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = vec![[1.0f64, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let rmsd = baseline_rmsd(&a, &b);
        assert!((rmsd - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rmsd_baseline_f32_correctness() {
        let a = vec![[0.0f32, 0.0, 0.0], [3.0, 4.0, 0.0]];
        let b = vec![[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let rmsd = baseline_rmsd(&a, &b);
        let expected = 12.5f32.sqrt();
        assert!((rmsd - expected).abs() < 1e-5);
    }

    #[test]
    fn kabsch_baseline_identity_rotation() {
        let a = vec![[1.0f32, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]];
        let (rot, rmsd) = baseline_kabsch(&a, &a);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (r, i) in rot.iter().zip(identity.iter()) {
            assert!((r - i).abs() < 1e-10);
        }
        assert!(rmsd < 1e-10);
    }

    #[test]
    fn umeyama_baseline_identity() {
        let a = vec![[1.0f32, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]];
        let (rot, scale, rmsd) = baseline_umeyama(&a, &a);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (r, i) in rot.iter().zip(identity.iter()) {
            assert!((r - i).abs() < 1e-10);
        }
        assert!((scale - 1.0).abs() < 1e-10);
        assert!(rmsd < 1e-10);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_rmsd, bench_kabsch, bench_umeyama
}
criterion_main!(benches);

// endregion
