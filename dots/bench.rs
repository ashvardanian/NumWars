//! Benchmark for matrix multiplication (GEMM) operations
//!
//! Tests F32, F64, BF16, F16, I8, U8, E4M3, E5M2, E2M3, E3M2 matrix multiplications
//! across multiple libraries:
//! - NumKong (single or multi-threaded via NUMWARS_THREADS)
//! - matrixmultiply (sgemm/dgemm, threaded via MATMUL_NUM_THREADS)
//! - ndarray (OpenBLAS-backed, respects NUMWARS_THREADS via OPENBLAS_NUM_THREADS)
//! - nalgebra (uses matrixmultiply, threaded via MATMUL_NUM_THREADS)
//! - faer (single or multi-threaded via NUMWARS_THREADS)
//!
//! Run with:
//! ```bash
//! # Run all benchmarks
//! cargo bench --features bench_dots --bench bench_dots
//!
//! # Configure matrix dimensions (m×n×k where C=A@B.T, C is m×n, A is m×k, B is n×k)
//! NUMWARS_DIMS_WIDTH=2048 NUMWARS_DIMS_HEIGHT=1024 NUMWARS_DIMS_DEPTH=512 \
//! cargo bench --features bench_dots
//!
//! # Filter to specific benchmarks
//! NUMWARS_FILTER="f32" cargo bench --features bench_dots
//! NUMWARS_FILTER="dots/f32" cargo bench --features bench_dots
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS_WIDTH: Matrix C width n (default: 2048)
//! - NUMWARS_DIMS_HEIGHT: Matrix C height m (default: 2048)
//! - NUMWARS_DIMS_DEPTH: Shared dimension k (default: 2048)
//! - NUMWARS_FILTER: Regex to filter benchmark names (default: none, runs all)
//! - NUMWARS_WARMUP_SECONDS: Warmup duration (default: 3.0)
//! - NUMWARS_PROFILE_SECONDS: Measurement duration (default: 10.0)
//!
//! Benchmark naming: dots/{dtype}/{height}x{width}x{depth}
//! Examples: dots/f32/2048x2048x2048, dots/i8/2048x2048x2048

extern crate blas_src;
extern crate openblas_src;

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use numkong::{bf16, capabilities, e2m3, e3m2, e4m3, e5m2, f16, Dots, PackedMatrix, Tensor};
use std::hint::black_box;
use utils::*;

/// Resolve the parallelism setting for faer based on NUMWARS_THREADS.
fn faer_parallelism() -> faer::Par {
    let threads = get_thread_count();
    if threads <= 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(threads)
    }
}

// region: matrixmultiply wrapper trait

/// The `matrixmultiply` crate doesn't use traits, so we add a thin wrapper.
/// https://docs.rs/matrixmultiply/latest/matrixmultiply/all.html
trait WrapMatrixMultiply: Sized {
    fn matmul(m: usize, k: usize, n: usize, a: &[Self], b: &[Self], c: &mut [Self]);
}

impl WrapMatrixMultiply for f32 {
    #[rustfmt::skip]
    fn matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        unsafe { matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), 1, k as isize, 0.0, c.as_mut_ptr(), n as isize, 1) };
    }
}

impl WrapMatrixMultiply for f64 {
    #[rustfmt::skip]
    fn matmul(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        unsafe { matrixmultiply::dgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), 1, k as isize, 0.0, c.as_mut_ptr(), n as isize, 1) };
    }
}

// endregion

// region: Per-library Run traits

trait RunNumKong: Dots + Clone + Send + Sync + Sized
where
    Self::Accumulator: Clone + Default + Send + Sync,
{
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: Self) {
        let threads = get_thread_count();
        let a = Tensor::<Self>::try_full(&[m, k], v.clone()).expect("Failed to allocate A");
        let b = Tensor::<Self>::try_full(&[n, k], v).expect("Failed to allocate B");
        let packed_b = PackedMatrix::try_pack(&b).expect("Failed to pack B");
        let mut c_out = Tensor::<Self::Accumulator>::try_zeros(&[m, n]).expect("Failed to allocate C");
        if threads > 1 {
            let mut pool = fork_union::ThreadPool::try_spawn(threads).expect("Failed to spawn thread pool");
            pool.for_threads(|_, _| {
                capabilities::configure_thread();
            })
            .join();
            group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    a.try_dots_packed_parallel_into(&packed_b, &mut c_out, &mut pool)
                        .expect("parallel matmul failed");
                })
            });
        } else {
            group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    a.try_dots_packed_into(&packed_b, &mut c_out).expect("matmul failed");
                })
            });
        }
    }
}
impl RunNumKong for f32 {}
impl RunNumKong for f64 {}
impl RunNumKong for i8 {}
impl RunNumKong for u8 {}
impl RunNumKong for bf16 {}
impl RunNumKong for f16 {}
impl RunNumKong for e4m3 {}
impl RunNumKong for e5m2 {}
impl RunNumKong for e2m3 {}
impl RunNumKong for e3m2 {}

trait RunMatrixMultiply: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self) {}
}
impl RunMatrixMultiply for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f32) {
        let a = vec![v; m * k];
        let b = vec![v; n * k];
        let mut c_out = vec![0.0f32; m * n];
        group.bench_function("matrixmultiply", |bench| {
            bench.iter(|| {
                f32::matmul(m, k, n, &a, &b, &mut c_out);
                black_box(&c_out);
            })
        });
    }
}
impl RunMatrixMultiply for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f64) {
        let a = vec![v; m * k];
        let b = vec![v; n * k];
        let mut c_out = vec![0.0f64; m * n];
        group.bench_function("matrixmultiply", |bench| {
            bench.iter(|| {
                f64::matmul(m, k, n, &a, &b, &mut c_out);
                black_box(&c_out);
            })
        });
    }
}
impl RunMatrixMultiply for i8 {}
impl RunMatrixMultiply for u8 {}
impl RunMatrixMultiply for bf16 {}
impl RunMatrixMultiply for f16 {}
impl RunMatrixMultiply for e4m3 {}
impl RunMatrixMultiply for e5m2 {}
impl RunMatrixMultiply for e2m3 {}
impl RunMatrixMultiply for e3m2 {}

trait RunNdarray: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self) {}
}
impl RunNdarray for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f32) {
        let a = ndarray::Array2::<f32>::from_shape_fn((m, k), |_| v);
        let b = ndarray::Array2::<f32>::from_shape_fn((n, k), |_| v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&b.t()))));
    }
}
impl RunNdarray for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f64) {
        let a = ndarray::Array2::<f64>::from_shape_fn((m, k), |_| v);
        let b = ndarray::Array2::<f64>::from_shape_fn((n, k), |_| v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&b.t()))));
    }
}
impl RunNdarray for i8 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: i8) {
        let a = ndarray::Array2::<i8>::from_shape_fn((m, k), |_| v);
        let b = ndarray::Array2::<i8>::from_shape_fn((n, k), |_| v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&b.t()))));
    }
}
impl RunNdarray for u8 {}
impl RunNdarray for bf16 {}
impl RunNdarray for f16 {}
impl RunNdarray for e4m3 {}
impl RunNdarray for e5m2 {}
impl RunNdarray for e2m3 {}
impl RunNdarray for e3m2 {}

trait RunNalgebra: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self) {}
}
impl RunNalgebra for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f32) {
        let a = nalgebra::DMatrix::<f32>::from_fn(m, k, |_, _| v);
        let b = nalgebra::DMatrix::<f32>::from_fn(n, k, |_, _| v);
        let bt = b.transpose();
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a * &bt)));
    }
}
impl RunNalgebra for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f64) {
        let a = nalgebra::DMatrix::<f64>::from_fn(m, k, |_, _| v);
        let b = nalgebra::DMatrix::<f64>::from_fn(n, k, |_, _| v);
        let bt = b.transpose();
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a * &bt)));
    }
}
impl RunNalgebra for i8 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: i8) {
        let a = nalgebra::DMatrix::<i8>::from_fn(m, k, |_, _| v);
        let b = nalgebra::DMatrix::<i8>::from_fn(n, k, |_, _| v);
        let bt = b.transpose();
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a * &bt)));
    }
}
impl RunNalgebra for u8 {}
impl RunNalgebra for bf16 {}
impl RunNalgebra for f16 {}
impl RunNalgebra for e4m3 {}
impl RunNalgebra for e5m2 {}
impl RunNalgebra for e2m3 {}
impl RunNalgebra for e3m2 {}

trait RunFaer: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self) {}
}
impl RunFaer for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f32) {
        let par = faer_parallelism();
        let a = faer::Mat::<f32>::from_fn(m, k, |_, _| v);
        let b = faer::Mat::<f32>::from_fn(n, k, |_, _| v);
        let mut c_out = faer::Mat::<f32>::zeros(m, n);
        group.bench_function("faer", |bench| {
            bench.iter(|| {
                faer::linalg::matmul::matmul(
                    &mut c_out,
                    faer::Accum::Replace,
                    a.as_ref(),
                    b.as_ref().transpose(),
                    faer::traits::math_utils::one::<f32>(),
                    par,
                );
                black_box(&c_out);
            })
        });
    }
}
impl RunFaer for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f64) {
        let par = faer_parallelism();
        let a = faer::Mat::<f64>::from_fn(m, k, |_, _| v);
        let b = faer::Mat::<f64>::from_fn(n, k, |_, _| v);
        let mut c_out = faer::Mat::<f64>::zeros(m, n);
        group.bench_function("faer", |bench| {
            bench.iter(|| {
                faer::linalg::matmul::matmul(
                    &mut c_out,
                    faer::Accum::Replace,
                    a.as_ref(),
                    b.as_ref().transpose(),
                    faer::traits::math_utils::one::<f64>(),
                    par,
                );
                black_box(&c_out);
            })
        });
    }
}
impl RunFaer for i8 {}
impl RunFaer for u8 {}
impl RunFaer for bf16 {}
impl RunFaer for f16 {}
impl RunFaer for e4m3 {}
impl RunFaer for e5m2 {}
impl RunFaer for e2m3 {}
impl RunFaer for e3m2 {}

// endregion

// region: Generic helper and entry point

fn bench_dtype<T>(c: &mut Criterion, dtype: &str, init: T)
where
    T: Dots + Clone + Send + Sync + RunNumKong + RunMatrixMultiply + RunNdarray + RunNalgebra + RunFaer,
    T::Accumulator: Clone + Default + Send + Sync,
{
    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let name = format!("dots/{}/{}x{}x{}", dtype, m, n, k);
    if !should_run_benchmark(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));

    <T as RunNumKong>::run(&mut group, m, n, k, init.clone());
    <T as RunMatrixMultiply>::run(&mut group, m, n, k, init.clone());
    <T as RunNdarray>::run(&mut group, m, n, k, init.clone());
    <T as RunNalgebra>::run(&mut group, m, n, k, init.clone());
    <T as RunFaer>::run(&mut group, m, n, k, init);

    group.finish();
}

extern "C" {
    fn openblas_set_num_threads(num_threads: std::ffi::c_int);
}

/// Propagate NUMWARS_THREADS to competitor backends at runtime.
///
/// OpenBLAS ignores env vars set after library init, so we call the C API directly.
/// matrixmultiply reads MATMUL_NUM_THREADS lazily on first use, so env var works.
fn propagate_thread_count() {
    let threads = get_thread_count();
    unsafe { openblas_set_num_threads(threads as std::ffi::c_int) };
    std::env::set_var("MATMUL_NUM_THREADS", threads.to_string());
}

fn bench_dots(c: &mut Criterion) {
    capabilities::configure_thread();
    propagate_thread_count();
    bench_dtype(c, "f32", 1.0f32);
    bench_dtype(c, "f64", 1.0f64);
    bench_dtype(c, "i8", 1i8);
    bench_dtype(c, "u8", 1u8);
    bench_dtype(c, "bf16", bf16::from_f32(1.0));
    bench_dtype(c, "f16", f16::from_f32(1.0));
    bench_dtype(c, "e4m3", e4m3::from_f32(1.0));
    bench_dtype(c, "e5m2", e5m2::from_f32(1.0));
    bench_dtype(c, "e2m3", e2m3::from_f32(1.0));
    bench_dtype(c, "e3m2", e3m2::from_f32(1.0));
}

// endregion

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_dots
}
criterion_main!(benches);
