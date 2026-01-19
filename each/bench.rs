//! Benchmark for elementwise tensor operations
//!
//! Compares NumKong Tensor API vs baseline Rust implementations and ndarray for
//! elementwise operations like addition, multiplication, scaling, and fused operations.
//!
//! Run with:
//! ```bash
//! # Run all benchmarks
//! cargo bench --features bench_each --bench bench_each
//!
//! # Configure problem size
//! NUMWARS_DIMS=1000000 cargo bench --features bench_each
//!
//! # Filter to specific benchmarks via regex
//! NUMWARS_FILTER="f32" cargo bench --features bench_each  # Only f32 dtypes
//! NUMWARS_FILTER="add" cargo bench --features bench_each  # Only add operations
//! NUMWARS_FILTER="each/add/f32" cargo bench --features bench_each  # Specific benchmark
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Tensor size in elements (default: 1000000)
//! - NUMWARS_FILTER: Regex to filter benchmark names (default: none, runs all)
//! - NUMWARS_WARMUP_SECONDS: Warmup time in seconds (default: 3.0)
//! - NUMWARS_PROFILE_SECONDS: Measurement time in seconds (default: 10.0)
//!
//! Benchmark naming: each/{operation}/{dtype}
//! Examples: each/add/f32, each/multiply/f64, each/scale/i8

#![allow(unused)]

#[path = "../utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use utils::*;

// Note: Configuration helpers (get_tensor_dims, etc.) are now in utils.rs.
// Filtering is done exclusively via should_run_benchmark() with hierarchical
// names like "each/operation/dtype".

// region: Baseline Implementations

/// Baseline elementwise addition
fn baseline_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Baseline elementwise addition (in-place)
fn baseline_add_inplace_f32(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y);
}

/// Baseline elementwise multiplication
fn baseline_multiply_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Baseline elementwise multiplication (in-place)
fn baseline_multiply_inplace_f32(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x *= y);
}

/// Baseline scalar multiplication
fn baseline_scale_f32(a: &[f32], scalar: f32) -> Vec<f32> {
    a.iter().map(|x| x * scalar).collect()
}

/// Baseline scalar multiplication (in-place)
fn baseline_scale_inplace_f32(a: &mut [f32], scalar: f32) {
    a.iter_mut().for_each(|x| *x *= scalar);
}

/// Baseline weighted sum: R = alpha * A + beta * B
fn baseline_wsum_f32(a: &[f32], b: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| alpha * x + beta * y)
        .collect()
}

/// Baseline fused multiply-add: R = alpha * A * B + beta * C
fn baseline_fma_f32(a: &[f32], b: &[f32], c: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((x, y), z)| alpha * x * y + beta * z)
        .collect()
}

// endregion

// region: Benchmarks

/// Benchmark elementwise addition
pub fn bench_add(c: &mut Criterion) {
    let size = get_tensor_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("each/add");
    group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));

    // f32
    if should_run_benchmark("each/add/f32") {
        let a = generate_random_f32(&mut rng, size);
        let b = generate_random_f32(&mut rng, size);

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_add_f32(&a, &b)))
        });

        // In-place version
        let mut a_mut = a.clone();
        group.bench_function("baseline_inplace_f32", |bench| {
            bench.iter(|| {
                baseline_add_inplace_f32(&mut a_mut, &b);
                black_box(&a_mut)
            })
        });

        #[cfg(feature = "ndarray")]
        {
            use ndarray::Array1;
            let a_nd = Array1::from(a.clone());
            let b_nd = Array1::from(b.clone());

            group.bench_function("ndarray_f32", |bench| {
                bench.iter(|| black_box(&a_nd + &b_nd))
            });
        }
    }

    group.finish();
}

/// Benchmark elementwise multiplication
pub fn bench_multiply(c: &mut Criterion) {
    let size = get_tensor_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("each/multiply");
    group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));

    // f32
    if should_run_benchmark("each/multiply/f32") {
        let a = generate_random_f32(&mut rng, size);
        let b = generate_random_f32(&mut rng, size);

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_multiply_f32(&a, &b)))
        });

        // In-place version
        let mut a_mut = a.clone();
        group.bench_function("baseline_inplace_f32", |bench| {
            bench.iter(|| {
                baseline_multiply_inplace_f32(&mut a_mut, &b);
                black_box(&a_mut)
            })
        });

        #[cfg(feature = "ndarray")]
        {
            use ndarray::Array1;
            let a_nd = Array1::from(a.clone());
            let b_nd = Array1::from(b.clone());

            group.bench_function("ndarray_f32", |bench| {
                bench.iter(|| black_box(&a_nd * &b_nd))
            });
        }
    }

    group.finish();
}

/// Benchmark scalar multiplication
pub fn bench_scale(c: &mut Criterion) {
    let size = get_tensor_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("each/scale");
    group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));

    // f32
    if should_run_benchmark("each/scale/f32") {
        let a = generate_random_f32(&mut rng, size);
        let scalar = 2.5f32;

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_scale_f32(&a, scalar)))
        });

        // In-place version
        let mut a_mut = a.clone();
        group.bench_function("baseline_inplace_f32", |bench| {
            bench.iter(|| {
                baseline_scale_inplace_f32(&mut a_mut, scalar);
                black_box(&a_mut)
            })
        });

        #[cfg(feature = "ndarray")]
        {
            use ndarray::Array1;
            let a_nd = Array1::from(a.clone());

            group.bench_function("ndarray_f32", |bench| {
                bench.iter(|| black_box(&a_nd * scalar))
            });
        }
    }

    group.finish();
}

/// Benchmark weighted sum: R = alpha * A + beta * B
pub fn bench_wsum(c: &mut Criterion) {
    let size = get_tensor_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("each/wsum");
    group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));

    // f32
    if should_run_benchmark("each/wsum/f32") {
        let a = generate_random_f32(&mut rng, size);
        let b = generate_random_f32(&mut rng, size);
        let alpha = 1.5f32;
        let beta = 2.5f32;

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_wsum_f32(&a, &b, alpha, beta)))
        });

        #[cfg(feature = "ndarray")]
        {
            use ndarray::Array1;
            let a_nd = Array1::from(a.clone());
            let b_nd = Array1::from(b.clone());

            group.bench_function("ndarray_f32", |bench| {
                bench.iter(|| black_box(&a_nd * alpha + &b_nd * beta))
            });
        }
    }

    group.finish();
}

/// Benchmark fused multiply-add: R = alpha * A * B + beta * C
pub fn bench_fma(c: &mut Criterion) {
    let size = get_tensor_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("each/fma");
    group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));

    // f32
    if should_run_benchmark("each/fma/f32") {
        let a = generate_random_f32(&mut rng, size);
        let b = generate_random_f32(&mut rng, size);
        let c = generate_random_f32(&mut rng, size);
        let alpha = 1.5f32;
        let beta = 2.5f32;

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_fma_f32(&a, &b, &c, alpha, beta)))
        });

        #[cfg(feature = "ndarray")]
        {
            use ndarray::Array1;
            let a_nd = Array1::from(a.clone());
            let b_nd = Array1::from(b.clone());
            let c_nd = Array1::from(c.clone());

            group.bench_function("ndarray_f32", |bench| {
                bench.iter(|| black_box(&a_nd * &b_nd * alpha + &c_nd * beta))
            });
        }
    }

    group.finish();
}

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets =
        bench_add,
        bench_multiply,
        bench_scale,
        bench_wsum,
        bench_fma
}
criterion_main!(benches);

// endregion
