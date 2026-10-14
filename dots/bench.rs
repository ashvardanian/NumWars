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
    let settings = Settings::read();
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    propagate_thread_count(settings.threads);
    let mut criterion = numwars::configure_criterion(&settings);
    bench_dots(&mut criterion, &settings);
    criterion.final_summary();
}
