//! Benchmark for matrix multiplication (GEMM) operations
//!
//! Tests F32, F64, BF16, F16, I8, U8, E4M3, E5M2, E2M3, E3M2 matrix multiplications, plus the
//! NumKong-only packed E2M1, I4, U4 and U1 ones, across multiple libraries:
//! - NumKong (single or multi-threaded via NUMWARS_THREADS)
//! - matrixmultiply (sgemm/dgemm, threaded via MATMUL_NUM_THREADS)
//! - ndarray (OpenBLAS-backed, respects NUMWARS_THREADS via OPENBLAS_NUM_THREADS)
//! - nalgebra (uses matrixmultiply, threaded via MATMUL_NUM_THREADS)
//! - faer (single or multi-threaded via NUMWARS_THREADS)
//!
//! Every dtype also gets a SYRK group, C = A × Aᵀ over the NUMWARS_DIMS_HEIGHT rows of A, beside
//! its GEMM one. NumKong's `dots_symmetric` fills the upper triangle and faer's triangular matmul
//! the lower one, while the other competitors compute the full product. Its throughput counts
//! only the n(n+1)/2 pairs of one triangle, so a full product earns no credit for the mirrored
//! half.
//!
//! A `cuda` build adds GEMM and SYRK groups for each GPU in NUMWARS_DEVICES, named like
//! `dots/bf16/16384x16384x16384<cuda:0>`, for every dtype NumKong has CUDA kernels for. Their
//! operands are random and live in unified memory, every round synchronizes the stream, and the
//! first round is checked against NumKong's CPU kernel on the leading rows. The rival is cuBLASLt
//! for the dtypes it multiplies, with cuBLAS's own SYRK for f32 and f64.
//!
//! Run with:
//! ```bash
//! # Run all benchmarks
//! cargo bench --features bench_dots --bench bench_dots
//!
//! # Configure matrix dimensions (m×n×k where C = A × Bᵀ, C is m×n, A is m×k, B is n×k)
//! NUMWARS_DIMS_WIDTH=2048 NUMWARS_DIMS_HEIGHT=1024 NUMWARS_DIMS_DEPTH=512 \
//! cargo bench --features bench_dots
//!
//! # Filter to specific benchmarks
//! NUMWARS_FILTER="f32" cargo bench --features bench_dots
//! NUMWARS_FILTER="dots/f32" cargo bench --features bench_dots
//! NUMWARS_FILTER="dots/symmetric" cargo bench --features bench_dots
//! ```
//!
//! Benchmark naming: dots/{dtype}/{height}x{width}x{depth} and dots/symmetric/{dtype}/{height}x{depth}
//! Examples: dots/f32/1024x128x1536, dots/i8/1024x128x1536, dots/symmetric/f32/1024x1536

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use faer::linalg::matmul::triangular::BlockStructure;
use numkong::prelude::*;
use numkong::{
    bf16, e2m1x2, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2, Capabilities, Dots, StorageElement, SymmetricDotsOps,
};

use numwars::{pack_dots_matrix, propagate_thread_count, try_spawn_pool, Settings};

#[cfg(feature = "cuda")]
use cudarc::cublas::{result as cublas, sys as cublas_sys};
#[cfg(feature = "cuda")]
use numkong::UnifiedAllocator;
#[cfg(feature = "cuda")]
use numwars::{gpu_check, gpu_devices, CublasLtGemm, CublasLtLayout, GpuDevice, GpuMetric, GpuScalar};

/// Resolve the parallelism setting for faer from the thread count.
fn faer_parallelism(threads: usize) -> faer::Par {
    if threads <= 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(threads)
    }
}

/// Fills the lower triangle of vectors × vectorsᵀ, faer's SYRK.
fn faer_symmetric<T: faer::traits::ComplexField>(
    gram: faer::MatMut<'_, T>,
    vectors: faer::MatRef<'_, T>,
    par: faer::Par,
) {
    faer::linalg::matmul::triangular::matmul(
        gram,
        BlockStructure::TriangularLower,
        faer::Accum::Replace,
        vectors,
        BlockStructure::Rectangular,
        vectors.transpose(),
        BlockStructure::Rectangular,
        faer::traits::math_utils::one::<T>(),
        par,
    );
}

// region: MatrixMultiply Wrapper Trait

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

// region: Per-Library Run Traits

trait RunNumKong: Dots + Clone + Send + Sync + Sized
where
    Self::Accumulator: Clone + Default + Send + Sync + 'static,
{
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: Self, threads: usize) {
        let a = Tensor::<Self>::full(&[m, k], v.clone()).expect("Failed to allocate A");
        let b = Tensor::<Self>::full(&[n, k], v).expect("Failed to allocate B");
        let mut c_out = Tensor::<Self::Accumulator>::zeros(&[m, n]).expect("Failed to allocate C");
        match try_spawn_pool(threads) {
            Some(mut pool) => {
                let packed_b = pack_dots_matrix(&b, Some(&mut pool));
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        a.dots_packed_parallel_into(&packed_b, &mut c_out, &mut pool)
                            .expect("parallel matmul failed");
                    })
                });
            }
            None => {
                let packed_b = pack_dots_matrix(&b, None);
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        a.dots_packed_into(&packed_b, &mut c_out).expect("matmul failed");
                    })
                });
            }
        }
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: Self, threads: usize) {
        let a = Tensor::<Self>::full(&[n, k], v).expect("Failed to allocate A");
        let mut c_out = Tensor::<Self::Accumulator>::zeros(&[n, n]).expect("Failed to allocate C");
        match try_spawn_pool(threads) {
            Some(mut pool) => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        a.dots_symmetric_parallel_into(&mut c_out, &mut pool)
                            .expect("parallel symmetric matmul failed");
                    })
                });
            }
            None => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        a.dots_symmetric_into(&mut c_out).expect("symmetric matmul failed");
                    })
                });
            }
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
impl RunNumKong for e2m1x2 {}
impl RunNumKong for i4x2 {}
impl RunNumKong for u4x2 {}
impl RunNumKong for u1x8 {}

trait RunMatrixMultiply: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _n: usize, _k: usize, _v: Self) {}
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
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f32) {
        let a = vec![v; n * k];
        let mut c_out = vec![0.0f32; n * n];
        group.bench_function("matrixmultiply", |bench| {
            bench.iter(|| {
                f32::matmul(n, k, n, &a, &a, &mut c_out);
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
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f64) {
        let a = vec![v; n * k];
        let mut c_out = vec![0.0f64; n * n];
        group.bench_function("matrixmultiply", |bench| {
            bench.iter(|| {
                f64::matmul(n, k, n, &a, &a, &mut c_out);
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
impl RunMatrixMultiply for e2m1x2 {}
impl RunMatrixMultiply for i4x2 {}
impl RunMatrixMultiply for u4x2 {}
impl RunMatrixMultiply for u1x8 {}

trait RunNdarray: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _n: usize, _k: usize, _v: Self) {}
}
impl RunNdarray for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f32) {
        let a = ndarray::Array2::<f32>::from_shape_fn((m, k), |_| v);
        let b = ndarray::Array2::<f32>::from_shape_fn((n, k), |_| v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&b.t()))));
    }
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f32) {
        let a = ndarray::Array2::<f32>::from_shape_fn((n, k), |_| v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&a.t()))));
    }
}
impl RunNdarray for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f64) {
        let a = ndarray::Array2::<f64>::from_shape_fn((m, k), |_| v);
        let b = ndarray::Array2::<f64>::from_shape_fn((n, k), |_| v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&b.t()))));
    }
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f64) {
        let a = ndarray::Array2::<f64>::from_shape_fn((n, k), |_| v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&a.t()))));
    }
}
// No SYRK row: the i8 product wraps in i8, so it is not the i32 matrix NumKong returns
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
impl RunNdarray for e2m1x2 {}
impl RunNdarray for i4x2 {}
impl RunNdarray for u4x2 {}
impl RunNdarray for u1x8 {}

trait RunNalgebra: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _n: usize, _k: usize, _v: Self) {}
}
impl RunNalgebra for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f32) {
        let a = nalgebra::DMatrix::<f32>::from_fn(m, k, |_, _| v);
        let b = nalgebra::DMatrix::<f32>::from_fn(n, k, |_, _| v);
        let bt = b.transpose();
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a * &bt)));
    }
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f32) {
        let a = nalgebra::DMatrix::<f32>::from_fn(n, k, |_, _| v);
        let at = a.transpose();
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a * &at)));
    }
}
impl RunNalgebra for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f64) {
        let a = nalgebra::DMatrix::<f64>::from_fn(m, k, |_, _| v);
        let b = nalgebra::DMatrix::<f64>::from_fn(n, k, |_, _| v);
        let bt = b.transpose();
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a * &bt)));
    }
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f64) {
        let a = nalgebra::DMatrix::<f64>::from_fn(n, k, |_, _| v);
        let at = a.transpose();
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a * &at)));
    }
}
// No SYRK row: the i8 product wraps in i8, so it is not the i32 matrix NumKong returns
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
impl RunNalgebra for e2m1x2 {}
impl RunNalgebra for i4x2 {}
impl RunNalgebra for u4x2 {}
impl RunNalgebra for u1x8 {}

trait RunFaer: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _m: usize, _n: usize, _k: usize, _v: Self, _par: faer::Par) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _n: usize, _k: usize, _v: Self, _par: faer::Par) {}
}
impl RunFaer for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f32, par: faer::Par) {
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
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f32, par: faer::Par) {
        let a = faer::Mat::<f32>::from_fn(n, k, |_, _| v);
        let mut c_out = faer::Mat::<f32>::zeros(n, n);
        group.bench_function("faer", |bench| {
            bench.iter(|| {
                faer_symmetric(c_out.as_mut(), a.as_ref(), par);
                black_box(&c_out);
            })
        });
    }
}
impl RunFaer for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, m: usize, n: usize, k: usize, v: f64, par: faer::Par) {
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
    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, n: usize, k: usize, v: f64, par: faer::Par) {
        let a = faer::Mat::<f64>::from_fn(n, k, |_, _| v);
        let mut c_out = faer::Mat::<f64>::zeros(n, n);
        group.bench_function("faer", |bench| {
            bench.iter(|| {
                faer_symmetric(c_out.as_mut(), a.as_ref(), par);
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
impl RunFaer for e2m1x2 {}
impl RunFaer for i4x2 {}
impl RunFaer for u4x2 {}
impl RunFaer for u1x8 {}

// endregion

// region: Generic Helpers

fn bench_dtype<T>(c: &mut Criterion, settings: &Settings, dtype: &str, init: T)
where
    T: Dots + Clone + Send + Sync + RunNumKong + RunMatrixMultiply + RunNdarray + RunNalgebra + RunFaer,
    T::Accumulator: Clone + Default + Send + Sync + 'static,
{
    let (m, n, k) = (settings.dims_height, settings.dims_width, settings.dims_depth);
    let name = format!("dots/{}/{}x{}x{}", dtype, m, n, k);
    if settings.selects(&name) {
        let mut group = c.benchmark_group(name);
        group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));

        <T as RunNumKong>::run(&mut group, m, n, k, init.clone(), settings.threads);
        <T as RunMatrixMultiply>::run(&mut group, m, n, k, init.clone());
        <T as RunNdarray>::run(&mut group, m, n, k, init.clone());
        <T as RunNalgebra>::run(&mut group, m, n, k, init.clone());
        <T as RunFaer>::run(&mut group, m, n, k, init.clone(), faer_parallelism(settings.threads));

        group.finish();
    }

    let name = format!("dots/symmetric/{dtype}/{m}x{k}");
    if !settings.selects(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    let pairs = m as u64 * (m as u64 + 1) / 2;
    group.throughput(Throughput::Elements(2 * pairs * k as u64));

    <T as RunNumKong>::run_symmetric(&mut group, m, k, init.clone(), settings.threads);
    <T as RunMatrixMultiply>::run_symmetric(&mut group, m, k, init.clone());
    <T as RunNdarray>::run_symmetric(&mut group, m, k, init.clone());
    <T as RunNalgebra>::run_symmetric(&mut group, m, k, init.clone());
    <T as RunFaer>::run_symmetric(&mut group, m, k, init, faer_parallelism(settings.threads));

    group.finish();
}

fn bench_dots(c: &mut Criterion, settings: &Settings) {
    bench_dtype(c, settings, "f32", 1.0f32);
    bench_dtype(c, settings, "f64", 1.0f64);
    bench_dtype(c, settings, "i8", 1i8);
    bench_dtype(c, settings, "u8", 1u8);
    bench_dtype(c, settings, "bf16", bf16::from_f32(1.0));
    bench_dtype(c, settings, "f16", f16::from_f32(1.0));
    bench_dtype(c, settings, "e4m3", e4m3::from_f32(1.0));
    bench_dtype(c, settings, "e5m2", e5m2::from_f32(1.0));
    bench_dtype(c, settings, "e2m3", e2m3::from_f32(1.0));
    bench_dtype(c, settings, "e3m2", e3m2::from_f32(1.0));
    bench_dtype(c, settings, "e2m1", e2m1x2::one());
    bench_dtype(c, settings, "i4", i4x2::one());
    bench_dtype(c, settings, "u4", u4x2::one());
    bench_dtype(c, settings, "u1", u1x8::one());
}

// endregion

// region: GPU Rows

/// Leading rows of a GPU result its first round is checked on.
#[cfg(feature = "cuda")]
const CHECKED_ROWS: usize = 16;

/// Relative error a GPU row may show against the CPU, which sums the depth in another order.
#[cfg(feature = "cuda")]
const GPU_TOLERANCE: f64 = 1e-3;

/// cuBLAS's own SYRK over row-major `[count, depth]` vectors, whose lower column-major triangle
/// is the upper row-major one.
#[cfg(feature = "cuda")]
type CublasSyrk<Scalar> = unsafe fn(
    cublas_sys::cublasHandle_t,
    *const Scalar,
    *mut <Scalar as GpuScalar>::Product,
    usize,
    usize,
) -> cublas_sys::cublasStatus_t;

#[cfg(feature = "cuda")]
unsafe fn cublas_ssyrk(
    handle: cublas_sys::cublasHandle_t,
    vectors: *const f32,
    gram: *mut f32,
    count: usize,
    depth: usize,
) -> cublas_sys::cublasStatus_t {
    use cublas_sys::{cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, cublasOperation_t::CUBLAS_OP_T};
    let (count, depth) = (count as i64, depth as i64);
    unsafe {
        cublas_sys::cublasSsyrk_v2_64(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,
            count,
            depth,
            &1.0,
            vectors,
            depth,
            &0.0,
            gram,
            count,
        )
    }
}

#[cfg(feature = "cuda")]
unsafe fn cublas_dsyrk(
    handle: cublas_sys::cublasHandle_t,
    vectors: *const f64,
    gram: *mut f64,
    count: usize,
    depth: usize,
) -> cublas_sys::cublasStatus_t {
    use cublas_sys::{cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, cublasOperation_t::CUBLAS_OP_T};
    let (count, depth) = (count as i64, depth as i64);
    unsafe {
        cublas_sys::cublasDsyrk_v2_64(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,
            count,
            depth,
            &1.0,
            vectors,
            depth,
            &0.0,
            gram,
            count,
        )
    }
}

/// NumKong's CPU result for the leading rows of `a` against every row of `b`, the reference each
/// GPU row's first round must match.
#[cfg(feature = "cuda")]
fn cpu_leading_rows<T: GpuScalar>(
    a: &Tensor<T, UnifiedAllocator>,
    b: &Tensor<T, UnifiedAllocator>,
) -> Tensor<T::Accumulator> {
    let (rows, depth) = (a.shape()[0].min(CHECKED_ROWS), a.shape()[1]);
    let values = rows * depth / T::dimensions_per_value();
    let leading = Tensor::<T>::from_slice(&a.as_slice()[..values], &[rows, depth]).expect("Failed to copy rows");
    let packed = DotsPackedMatrix::new(b).expect("Failed to pack B on the CPU");
    leading.dots_packed(&packed).expect("CPU dots failed")
}

/// Adds the cuBLASLt product `a × bᵀ` to `group` as `cublaslt`, once its first round matches
/// `expected`, the full product of a SYRK group included.
#[cfg(feature = "cuda")]
fn bench_cublaslt<T: GpuScalar>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    device: &GpuDevice,
    a: &Tensor<T, UnifiedAllocator>,
    b: &Tensor<T, UnifiedAllocator>,
    expected: &[T::Accumulator],
    upper_only: bool,
) {
    let Some(types) = T::CUBLASLT else {
        return;
    };
    let (rows, columns, depth) = (a.shape()[0], b.shape()[0], a.shape()[1]);
    let gemm = match CublasLtGemm::new(device, types, CublasLtLayout::new(rows, columns, depth)) {
        Ok(gemm) => gemm,
        Err(error) => return println!("{name}/cublaslt is unavailable: {error}"),
    };
    let mut product =
        Tensor::<T::Product, _>::zeros_in(&[rows, columns], device.allocator()).expect("Failed to allocate");
    let launch = |product: &mut Tensor<T::Product, UnifiedAllocator>| {
        gemm.launch(a, b, product);
        device.synchronize();
    };
    launch(&mut product);
    match gpu_check(
        &format!("{name}/cublaslt"),
        product.as_slice(),
        expected,
        columns,
        upper_only,
        GPU_TOLERANCE,
    ) {
        Ok(()) => {
            group.bench_function("cublaslt", |bench| bench.iter(|| launch(&mut product)));
        }
        Err(error) => println!("Skipping a rival that {error}"),
    }
}

/// Adds cuBLAS's own SYRK of `vectors` to `group` as `cublas`, once its first round matches `expected`.
#[cfg(feature = "cuda")]
fn bench_cublas_syrk<T: GpuScalar>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    device: &GpuDevice,
    vectors: &Tensor<T, UnifiedAllocator>,
    expected: &[T::Accumulator],
    syrk: CublasSyrk<T>,
) {
    let (count, depth) = (vectors.shape()[0], vectors.shape()[1]);
    let handle = cublas::create_handle().expect("Failed to create a cuBLAS handle");
    unsafe { cublas::set_stream(handle, device.stream.cast()) }.expect("Failed to set the cuBLAS stream");
    let mut gram = Tensor::<T::Product, _>::zeros_in(&[count, count], device.allocator()).expect("Failed to allocate");
    let launch = |gram: &mut Tensor<T::Product, UnifiedAllocator>| {
        let status = unsafe { syrk(handle, vectors.as_ptr(), gram.as_mut_ptr(), count, depth) };
        assert_eq!(
            status,
            cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "cuBLAS SYRK failed"
        );
        device.synchronize();
    };
    launch(&mut gram);
    match gpu_check(
        &format!("{name}/cublas"),
        gram.as_slice(),
        expected,
        count,
        true,
        GPU_TOLERANCE,
    ) {
        Ok(()) => {
            group.bench_function("cublas", |bench| bench.iter(|| launch(&mut gram)));
        }
        Err(error) => println!("Skipping a rival that {error}"),
    }
    unsafe { cublas::destroy_handle(handle) }.expect("Failed to destroy the cuBLAS handle");
}

/// GEMM and SYRK groups of `T` on each GPU, over random operands in unified memory, each round
/// synchronized and each first round checked against NumKong's CPU kernel.
#[cfg(feature = "cuda")]
fn bench_dtype_gpu<T: GpuScalar>(
    c: &mut Criterion,
    settings: &Settings,
    devices: &[GpuDevice],
    dtype: &str,
    syrk: Option<CublasSyrk<T>>,
) {
    let (m, n, k) = (settings.dims_height, settings.dims_width, settings.dims_depth);
    let seed = u64::from(settings.seed);
    for device in devices {
        let name = format!("dots/{dtype}/{m}x{n}x{k}{}", device.label());
        if settings.selects(&name) {
            let a = device.random::<T>(&[m, k], seed);
            let b = device.random::<T>(&[n, k], seed + 1);
            let mut product =
                Tensor::<T::Accumulator, _>::zeros_in(&[m, n], device.allocator()).expect("Failed to allocate C");
            let Some(packed) = device.pack(&b) else {
                println!("{name} is unavailable: NumKong has no CUDA kernel for {dtype}");
                continue;
            };
            let launch = |product: &mut Tensor<T::Accumulator, UnifiedAllocator>| {
                device
                    .packed(GpuMetric::Dots, &a, &packed, product)
                    .expect("No packed CUDA kernel");
                device.synchronize();
            };
            launch(&mut product);
            let expected = cpu_leading_rows(&a, &b);
            gpu_check(&name, product.as_slice(), expected.as_slice(), n, false, GPU_TOLERANCE)
                .unwrap_or_else(|error| panic!("{error}"));
            let mut group = c.benchmark_group(&name);
            group.throughput(Throughput::Elements(2 * m as u64 * n as u64 * k as u64));
            group.bench_function("numkong", |bench| bench.iter(|| launch(&mut product)));
            bench_cublaslt(&mut group, &name, device, &a, &b, expected.as_slice(), false);
            group.finish();
        }

        let name = format!("dots/symmetric/{dtype}/{m}x{k}{}", device.label());
        if settings.selects(&name) {
            let a = device.random::<T>(&[m, k], seed);
            let mut gram =
                Tensor::<T::Accumulator, _>::zeros_in(&[m, m], device.allocator()).expect("Failed to allocate C");
            if device.symmetric(GpuMetric::Dots, &a, &mut gram).is_none() {
                println!("{name} is unavailable: NumKong has no CUDA kernel for {dtype}");
                continue;
            }
            device.synchronize();
            let expected = cpu_leading_rows(&a, &a);
            gpu_check(&name, gram.as_slice(), expected.as_slice(), m, true, GPU_TOLERANCE)
                .unwrap_or_else(|error| panic!("{error}"));
            let mut group = c.benchmark_group(&name);
            let pairs = m as u64 * (m as u64 + 1) / 2;
            group.throughput(Throughput::Elements(2 * pairs * k as u64));
            group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    device.symmetric(GpuMetric::Dots, &a, &mut gram);
                    device.synchronize();
                })
            });
            match syrk {
                Some(syrk) => bench_cublas_syrk(&mut group, &name, device, &a, expected.as_slice(), syrk),
                None => bench_cublaslt(&mut group, &name, device, &a, &a, expected.as_slice(), true),
            }
            group.finish();
        }
    }
}

/// Every dtype NumKong's CUDA kernels take, on each GPU.
#[cfg(feature = "cuda")]
fn bench_dots_gpu(c: &mut Criterion, settings: &Settings, devices: &[GpuDevice]) {
    bench_dtype_gpu::<f32>(c, settings, devices, "f32", Some(cublas_ssyrk));
    bench_dtype_gpu::<f64>(c, settings, devices, "f64", Some(cublas_dsyrk));
    bench_dtype_gpu::<i8>(c, settings, devices, "i8", None);
    bench_dtype_gpu::<u8>(c, settings, devices, "u8", None);
    bench_dtype_gpu::<bf16>(c, settings, devices, "bf16", None);
    bench_dtype_gpu::<f16>(c, settings, devices, "f16", None);
    bench_dtype_gpu::<e4m3>(c, settings, devices, "e4m3", None);
    bench_dtype_gpu::<e5m2>(c, settings, devices, "e5m2", None);
    bench_dtype_gpu::<e2m3>(c, settings, devices, "e2m3", None);
    bench_dtype_gpu::<e3m2>(c, settings, devices, "e3m2", None);
    bench_dtype_gpu::<e2m1x2>(c, settings, devices, "e2m1", None);
    bench_dtype_gpu::<i4x2>(c, settings, devices, "i4", None);
    bench_dtype_gpu::<u4x2>(c, settings, devices, "u4", None);
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn dots_symmetric_rows_match_serial_oracle() {
        Capabilities::cpu_enabled().configure_thread().unwrap();
        let (n, k) = (5, 16);
        // Small integers keep every product and sum exact in f32
        let vectors: Vec<f32> = (0..n * k).map(|index| (index * 7 % 11) as f32 - 5.0).collect();
        let expected =
            |i: usize, j: usize| -> f64 { (0..k).map(|d| vectors[i * k + d] * vectors[j * k + d]).sum::<f32>() as f64 };

        let numkong_vectors = Tensor::<f32>::from_slice(&vectors, &[n, k]).unwrap();
        let mut numkong_gram = Tensor::<f64>::zeros(&[n, n]).unwrap();
        numkong_vectors.dots_symmetric_into(&mut numkong_gram).unwrap();

        let mut matrixmultiply_gram = vec![0.0f32; n * n];
        f32::matmul(n, k, n, &vectors, &vectors, &mut matrixmultiply_gram);

        let ndarray_vectors = ndarray::Array2::from_shape_vec((n, k), vectors.clone()).unwrap();
        let ndarray_gram = ndarray_vectors.dot(&ndarray_vectors.t());

        let nalgebra_vectors = nalgebra::DMatrix::from_row_slice(n, k, &vectors);
        let nalgebra_gram = &nalgebra_vectors * &nalgebra_vectors.transpose();

        let faer_vectors = faer::Mat::<f32>::from_fn(n, k, |row, column| vectors[row * k + column]);
        let mut faer_gram = faer::Mat::<f32>::zeros(n, n);
        faer_symmetric(faer_gram.as_mut(), faer_vectors.as_ref(), faer::Par::Seq);

        for i in 0..n {
            for j in i..n {
                let expected = expected(i, j);
                assert_eq!(numkong_gram.as_slice()[i * n + j], expected, "numkong[{i}, {j}]");
                assert_eq!(matrixmultiply_gram[i * n + j] as f64, expected, "sgemm[{i}, {j}]");
                assert_eq!(ndarray_gram[(i, j)] as f64, expected, "ndarray[{i}, {j}]");
                assert_eq!(nalgebra_gram[(i, j)] as f64, expected, "nalgebra[{i}, {j}]");
                assert_eq!(faer_gram[(j, i)] as f64, expected, "faer[{j}, {i}]");
            }
        }
    }
}

// endregion

fn main() {
    let settings = Settings::read("dots");
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    propagate_thread_count(settings.threads);
    #[cfg(feature = "cuda")]
    let devices = gpu_devices(&settings);
    let mut criterion = numwars::configure_criterion(&settings);
    bench_dots(&mut criterion, &settings);
    #[cfg(feature = "cuda")]
    bench_dots_gpu(&mut criterion, &settings, &devices);
    criterion.final_summary();
}
