//! Benchmark for matrix multiplication (GEMM) operations
//!
//! Tests F32, F64, BF16, I8 matrix multiplications across multiple libraries:
//! - NumKong (multi-threaded parallel)
//! - ndarray (single-threaded)
//! - nalgebra (single-threaded)
//! - faer (single-threaded)
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
//! NUMWARS_FILTER="f32" cargo bench --features bench_dots  # Only f32 dtypes
//! NUMWARS_FILTER="dots/numkong/f32" cargo bench --features bench_dots
//! NUMWARS_FILTER="dots/ndarray" cargo bench --features bench_dots  # Only ndarray
//!
//! # Control thread count (NumKong only)
//! NUMWARS_THREADS=8 cargo bench --features bench_dots
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS_WIDTH: Matrix C width (n) (default: 1024)
//! - NUMWARS_DIMS_HEIGHT: Matrix C height (m) (default: 1024)
//! - NUMWARS_DIMS_DEPTH: Shared dimension (k) (default: 1024)
//! - NUMWARS_THREADS: Thread count for parallel benchmarks (default: num_cpus)
//! - NUMWARS_FILTER: Regex to filter benchmark names (default: none, runs all)
//! - NUMWARS_WARMUP_SECONDS: Warmup duration (default: 3.0)
//! - NUMWARS_PROFILE_SECONDS: Measurement duration (default: 10.0)
//!
//! Benchmark naming: dots/{library}/{dtype}/{m}x{n}x{k}/{threads}t
//! Examples: dots/numkong/f32/1024x1024x1024/8t, dots/ndarray/f32/1024x1024x1024/1t

#[path = "../utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numkong::{bf16, capabilities, MatrixMultiplier, Tensor};
use utils::*;

// region: NumKong Benchmarks

fn bench_numkong_f32(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = get_thread_count();

    let benchmark_name = format!("dots/numkong/f32/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    // Create thread pool once, outside all benchmarks
    let mut pool =
        fork_union::ThreadPool::try_spawn(threads).expect("Failed to create thread pool");

    // Allocate outside the timed region
    // C = A @ B.T where A is m×k, B is n×k, C is m×n
    let a = Tensor::<f32>::try_new(&[m, k], 1.0f32).expect("Failed to allocate A");
    let b = Tensor::<f32>::try_new(&[n, k], 1.0f32).expect("Failed to allocate B");
    let packed_b = MatrixMultiplier::try_pack(&b).expect("Failed to pack B");
    let mut c_out = Tensor::<f32>::try_new(&[m, n], 0.0f32).expect("Failed to allocate C");

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("numkong_parallel", |bench| {
        bench.iter(|| {
            a.try_matmul_parallel_into(&packed_b, &mut c_out, &mut pool)
                .expect("matmul failed");
        });
    });

    group.finish();
}

fn bench_numkong_f64(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = get_thread_count();

    let benchmark_name = format!("dots/numkong/f64/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    let mut pool =
        fork_union::ThreadPool::try_spawn(threads).expect("Failed to create thread pool");

    let a = Tensor::<f64>::try_new(&[m, k], 1.0f64).expect("Failed to allocate A");
    let b = Tensor::<f64>::try_new(&[n, k], 1.0f64).expect("Failed to allocate B");
    let packed_b = MatrixMultiplier::try_pack(&b).expect("Failed to pack B");
    let mut c_out = Tensor::<f64>::try_new(&[m, n], 0.0f64).expect("Failed to allocate C");

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("numkong_parallel", |bench| {
        bench.iter(|| {
            a.try_matmul_parallel_into(&packed_b, &mut c_out, &mut pool)
                .expect("matmul failed");
        });
    });

    group.finish();
}

fn bench_numkong_bf16(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = get_thread_count();

    let benchmark_name = format!("dots/numkong/bf16/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    let mut pool =
        fork_union::ThreadPool::try_spawn(threads).expect("Failed to create thread pool");

    let a = Tensor::<bf16>::try_new(&[m, k], bf16::from_f32(1.0)).expect("Failed to allocate A");
    let b = Tensor::<bf16>::try_new(&[n, k], bf16::from_f32(1.0)).expect("Failed to allocate B");
    let packed_b = MatrixMultiplier::try_pack(&b).expect("Failed to pack B");
    let mut c_out = Tensor::<f32>::try_new(&[m, n], 0.0f32).expect("Failed to allocate C");

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("numkong_parallel", |bench| {
        bench.iter(|| {
            a.try_matmul_parallel_into(&packed_b, &mut c_out, &mut pool)
                .expect("matmul failed");
        });
    });

    group.finish();
}

fn bench_numkong_i8(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = get_thread_count();

    let benchmark_name = format!("dots/numkong/i8/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    let mut pool =
        fork_union::ThreadPool::try_spawn(threads).expect("Failed to create thread pool");

    let a = Tensor::<i8>::try_new(&[m, k], 1i8).expect("Failed to allocate A");
    let b = Tensor::<i8>::try_new(&[n, k], 1i8).expect("Failed to allocate B");
    let packed_b = MatrixMultiplier::try_pack(&b).expect("Failed to pack B");
    let mut c_out = Tensor::<i32>::try_new(&[m, n], 0i32).expect("Failed to allocate C");

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("numkong_parallel", |bench| {
        bench.iter(|| {
            a.try_matmul_parallel_into(&packed_b, &mut c_out, &mut pool)
                .expect("matmul failed");
        });
    });

    group.finish();
}

// endregion

// region: ndarray Benchmarks

#[cfg(feature = "ndarray")]
fn bench_ndarray_f32(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1; // ndarray is single-threaded

    let benchmark_name = format!("dots/ndarray/f32/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use ndarray::Array2;
    let a = Array2::<f32>::from_shape_fn((m, k), |_| 1.0);
    let b = Array2::<f32>::from_shape_fn((n, k), |_| 1.0);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("ndarray", |bench| {
        bench.iter(|| {
            black_box(a.dot(&b.t()))
        });
    });

    group.finish();
}

#[cfg(feature = "ndarray")]
fn bench_ndarray_f64(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/ndarray/f64/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use ndarray::Array2;
    let a = Array2::<f64>::from_shape_fn((m, k), |_| 1.0);
    let b = Array2::<f64>::from_shape_fn((n, k), |_| 1.0);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("ndarray", |bench| {
        bench.iter(|| {
            black_box(a.dot(&b.t()))
        });
    });

    group.finish();
}

#[cfg(feature = "ndarray")]
fn bench_ndarray_bf16(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/ndarray/bf16/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use half::bf16;
    use ndarray::Array2;
    let a = Array2::<bf16>::from_shape_fn((m, k), |_| bf16::from_f32(1.0));
    let b = Array2::<bf16>::from_shape_fn((n, k), |_| bf16::from_f32(1.0));

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("ndarray", |bench| {
        bench.iter(|| {
            black_box(a.dot(&b.t()))
        });
    });

    group.finish();
}

#[cfg(feature = "ndarray")]
fn bench_ndarray_i8(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/ndarray/i8/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use ndarray::Array2;
    let a = Array2::<i8>::from_shape_fn((m, k), |_| 1);
    let b = Array2::<i8>::from_shape_fn((n, k), |_| 1);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("ndarray", |bench| {
        bench.iter(|| {
            black_box(a.dot(&b.t()))
        });
    });

    group.finish();
}

// endregion

// region: nalgebra Benchmarks

#[cfg(feature = "nalgebra")]
fn bench_nalgebra_f32(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/nalgebra/f32/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use nalgebra::DMatrix;
    let a = DMatrix::<f32>::from_fn(m, k, |_, _| 1.0);
    let b = DMatrix::<f32>::from_fn(n, k, |_, _| 1.0);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("nalgebra", |bench| {
        bench.iter(|| {
            black_box(&a * b.transpose())
        });
    });

    group.finish();
}

#[cfg(feature = "nalgebra")]
fn bench_nalgebra_f64(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/nalgebra/f64/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use nalgebra::DMatrix;
    let a = DMatrix::<f64>::from_fn(m, k, |_, _| 1.0);
    let b = DMatrix::<f64>::from_fn(n, k, |_, _| 1.0);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("nalgebra", |bench| {
        bench.iter(|| {
            black_box(&a * b.transpose())
        });
    });

    group.finish();
}

#[cfg(feature = "nalgebra")]
fn bench_nalgebra_bf16(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/nalgebra/bf16/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use half::bf16;
    use nalgebra::DMatrix;
    let a = DMatrix::<bf16>::from_fn(m, k, |_, _| bf16::from_f32(1.0));
    let b = DMatrix::<bf16>::from_fn(n, k, |_, _| bf16::from_f32(1.0));

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("nalgebra", |bench| {
        bench.iter(|| {
            black_box(&a * b.transpose())
        });
    });

    group.finish();
}

#[cfg(feature = "nalgebra")]
fn bench_nalgebra_i8(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/nalgebra/i8/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use nalgebra::DMatrix;
    let a = DMatrix::<i8>::from_fn(m, k, |_, _| 1);
    let b = DMatrix::<i8>::from_fn(n, k, |_, _| 1);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("nalgebra", |bench| {
        bench.iter(|| {
            black_box(&a * b.transpose())
        });
    });

    group.finish();
}

// endregion

// region: faer Benchmarks

#[cfg(feature = "faer")]
fn bench_faer_f32(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/faer/f32/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use faer::Mat;
    let a = Mat::<f32>::from_fn(m, k, |_, _| 1.0);
    let b = Mat::<f32>::from_fn(n, k, |_, _| 1.0);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("faer", |bench| {
        bench.iter(|| {
            black_box(faer::linalg::matmul::matmul(
                a.as_ref(),
                b.as_ref().transpose(),
                None,
                1.0,
                faer::Parallelism::None,
            ))
        });
    });

    group.finish();
}

#[cfg(feature = "faer")]
fn bench_faer_f64(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/faer/f64/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use faer::Mat;
    let a = Mat::<f64>::from_fn(m, k, |_, _| 1.0);
    let b = Mat::<f64>::from_fn(n, k, |_, _| 1.0);

    let ops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    group.throughput(Throughput::Elements(ops));

    group.bench_function("faer", |bench| {
        bench.iter(|| {
            black_box(faer::linalg::matmul::matmul(
                a.as_ref(),
                b.as_ref().transpose(),
                None,
                1.0,
                faer::Parallelism::None,
            ))
        });
    });

    group.finish();
}

#[cfg(feature = "faer")]
fn bench_faer_bf16(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/faer/bf16/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    use half::bf16;
    // Note: faer doesn't natively support bf16, skip or convert to f32
    // For now, we'll skip this benchmark
    println!("Skipping faer bf16 benchmark (not natively supported)");

    group.finish();
}

#[cfg(feature = "faer")]
fn bench_faer_i8(c: &mut Criterion) {
    capabilities::configure_thread();

    let m = get_matrix_dims_height();
    let n = get_matrix_dims_width();
    let k = get_matrix_dims_depth();
    let threads = 1;

    let benchmark_name = format!("dots/faer/i8/{}x{}x{}/{}t", m, n, k, threads);
    if !should_run_benchmark(&benchmark_name) {
        return;
    }

    let mut group = c.benchmark_group(benchmark_name);

    // Note: faer doesn't natively support i8, skip or convert
    println!("Skipping faer i8 benchmark (not natively supported)");

    group.finish();
}

// endregion

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets =
        bench_numkong_f32,
        bench_numkong_f64,
        bench_numkong_bf16,
        bench_numkong_i8,

        #[cfg(feature = "ndarray")]
        bench_ndarray_f32,
        #[cfg(feature = "ndarray")]
        bench_ndarray_f64,
        #[cfg(feature = "ndarray")]
        bench_ndarray_bf16,
        #[cfg(feature = "ndarray")]
        bench_ndarray_i8,

        #[cfg(feature = "nalgebra")]
        bench_nalgebra_f32,
        #[cfg(feature = "nalgebra")]
        bench_nalgebra_f64,
        #[cfg(feature = "nalgebra")]
        bench_nalgebra_bf16,
        #[cfg(feature = "nalgebra")]
        bench_nalgebra_i8,

        #[cfg(feature = "faer")]
        bench_faer_f32,
        #[cfg(feature = "faer")]
        bench_faer_f64,
        #[cfg(feature = "faer")]
        bench_faer_bf16,
        #[cfg(feature = "faer")]
        bench_faer_i8,
}
criterion_main!(benches);
