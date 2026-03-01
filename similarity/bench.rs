//! Benchmark for vector similarity operations
//!
//! Compares NumKong vs baseline Rust implementations for pairwise similarity metrics.
//! Benchmarks spatial (dot, angular, sqeuclidean), binary (hamming, jaccard), and probability
//! (Jensen-Shannon, Kullback-Leibler) distances.
//!
//! Run with:
//! ```bash
//! # Run all benchmarks
//! cargo bench --features bench_similarity --bench bench_similarity
//!
//! # Configure problem size
//! NUMWARS_DIMS=512 cargo bench --features bench_similarity
//!
//! # Filter to specific benchmarks via regex
//! NUMWARS_FILTER="f32" cargo bench --features bench_similarity  # Only f32 dtypes
//! NUMWARS_FILTER="angular|dot" cargo bench --features bench_similarity  # Only angular and dot
//! NUMWARS_FILTER="similarity/angular/f32" cargo bench --features bench_similarity  # Specific benchmark
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Vector dimensions (default: 1536)
//! - NUMWARS_BATCH_SIZE: Number of vector pairs (default: 1000)
//! - NUMWARS_FILTER: Regex to filter benchmark names (default: none, runs all)
//! - NUMWARS_WARMUP_SECONDS: Warmup time in seconds (default: 3.0)
//! - NUMWARS_PROFILE_SECONDS: Measurement time in seconds (default: 10.0)
//!
//! Benchmark naming: similarity/{metric}/{dtype}
//! Examples: similarity/angular/f32, similarity/dot/f64, similarity/sqeuclidean/i8

#![allow(unused)]

#[path = "../utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_traits::{AsPrimitive, Num, NumCast};
use numkong::SpatialSimilarity as NumKongSpatial;
use rand::Rng;
use std::ops::AddAssign;
use utils::*;

// Note: Configuration helpers (get_vector_dims, get_batch_size, etc.) are now
// in utils.rs. Filtering is done exclusively via should_run_benchmark() with
// hierarchical names like "similarity/metric/dtype".

// region: Baseline Implementations

/// Baseline angular distance with 8-way unrolling
pub fn baseline_angular_unrolled<T, Acc>(a: &[T], b: &[T]) -> Option<f32>
where
    T: Num + Copy + NumCast + AsPrimitive<f32>,
    Acc: Num + Copy + NumCast + AddAssign + 'static,
    T: AsPrimitive<Acc>,
{
    if a.len() != b.len() {
        return None;
    }

    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc1 = Acc::zero();
    let mut acc2 = Acc::zero();
    let mut acc3 = Acc::zero();
    let mut acc4 = Acc::zero();
    let mut acc5 = Acc::zero();
    let mut acc6 = Acc::zero();
    let mut acc7 = Acc::zero();
    let mut acc8 = Acc::zero();

    let mut norm_a1 = Acc::zero();
    let mut norm_a2 = Acc::zero();
    let mut norm_b1 = Acc::zero();
    let mut norm_b2 = Acc::zero();

    while i < (a.len() - remainder) {
        unsafe {
            let a1 = *a.get_unchecked(i);
            let a2 = *a.get_unchecked(i + 1);
            let a3 = *a.get_unchecked(i + 2);
            let a4 = *a.get_unchecked(i + 3);
            let a5 = *a.get_unchecked(i + 4);
            let a6 = *a.get_unchecked(i + 5);
            let a7 = *a.get_unchecked(i + 6);
            let a8 = *a.get_unchecked(i + 7);

            let b1 = *b.get_unchecked(i);
            let b2 = *b.get_unchecked(i + 1);
            let b3 = *b.get_unchecked(i + 2);
            let b4 = *b.get_unchecked(i + 3);
            let b5 = *b.get_unchecked(i + 4);
            let b6 = *b.get_unchecked(i + 5);
            let b7 = *b.get_unchecked(i + 6);
            let b8 = *b.get_unchecked(i + 7);

            let a1_acc: Acc = NumCast::from(a1).unwrap();
            let a2_acc: Acc = NumCast::from(a2).unwrap();
            let a3_acc: Acc = NumCast::from(a3).unwrap();
            let a4_acc: Acc = NumCast::from(a4).unwrap();
            let a5_acc: Acc = NumCast::from(a5).unwrap();
            let a6_acc: Acc = NumCast::from(a6).unwrap();
            let a7_acc: Acc = NumCast::from(a7).unwrap();
            let a8_acc: Acc = NumCast::from(a8).unwrap();

            let b1_acc: Acc = NumCast::from(b1).unwrap();
            let b2_acc: Acc = NumCast::from(b2).unwrap();
            let b3_acc: Acc = NumCast::from(b3).unwrap();
            let b4_acc: Acc = NumCast::from(b4).unwrap();
            let b5_acc: Acc = NumCast::from(b5).unwrap();
            let b6_acc: Acc = NumCast::from(b6).unwrap();
            let b7_acc: Acc = NumCast::from(b7).unwrap();
            let b8_acc: Acc = NumCast::from(b8).unwrap();

            acc1 += a1_acc * b1_acc;
            acc2 += a2_acc * b2_acc;
            acc3 += a3_acc * b3_acc;
            acc4 += a4_acc * b4_acc;
            acc5 += a5_acc * b5_acc;
            acc6 += a6_acc * b6_acc;
            acc7 += a7_acc * b7_acc;
            acc8 += a8_acc * b8_acc;

            norm_a1 += a1_acc * a1_acc + a2_acc * a2_acc + a3_acc * a3_acc + a4_acc * a4_acc;
            norm_b1 += b1_acc * b1_acc + b2_acc * b2_acc + b3_acc * b3_acc + b4_acc * b4_acc;

            norm_a2 += a5_acc * a5_acc + a6_acc * a6_acc + a7_acc * a7_acc + a8_acc * a8_acc;
            norm_b2 += b5_acc * b5_acc + b6_acc * b6_acc + b7_acc * b7_acc + b8_acc * b8_acc;
        }

        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        unsafe {
            let a_acc: Acc = NumCast::from(*a.get_unchecked(i)).unwrap();
            let b_acc: Acc = NumCast::from(*b.get_unchecked(i)).unwrap();
            acc1 += a_acc * b_acc;
            norm_a1 += a_acc * a_acc;
            norm_b1 += b_acc * b_acc;
        }
        i += 1;
    }

    let dot_product = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8;
    let norm_a = norm_a1 + norm_a2;
    let norm_b = norm_b1 + norm_b2;

    let dot_product_f32: f32 = NumCast::from(dot_product).unwrap();
    let norm_a_f32: f32 = NumCast::from(norm_a).unwrap();
    let norm_b_f32: f32 = NumCast::from(norm_b).unwrap();

    Some(1.0 - (dot_product_f32 / (norm_a_f32.sqrt() * norm_b_f32.sqrt())))
}

/// Baseline squared Euclidean distance with 8-way unrolling
pub fn baseline_sqeuclidean_unrolled<T, Acc>(a: &[T], b: &[T]) -> Option<f32>
where
    T: Num + Copy + NumCast,
    Acc: Num + Copy + NumCast + AddAssign + 'static,
    T: AsPrimitive<Acc>,
{
    if a.len() != b.len() {
        return None;
    }
    let mut i = 0;
    let remainder = a.len() % 8;

    let mut acc1 = Acc::zero();
    let mut acc2 = Acc::zero();
    let mut acc3 = Acc::zero();
    let mut acc4 = Acc::zero();
    let mut acc5 = Acc::zero();
    let mut acc6 = Acc::zero();
    let mut acc7 = Acc::zero();
    let mut acc8 = Acc::zero();

    while i < (a.len() - remainder) {
        unsafe {
            let a1 = *a.get_unchecked(i);
            let a2 = *a.get_unchecked(i + 1);
            let a3 = *a.get_unchecked(i + 2);
            let a4 = *a.get_unchecked(i + 3);
            let a5 = *a.get_unchecked(i + 4);
            let a6 = *a.get_unchecked(i + 5);
            let a7 = *a.get_unchecked(i + 6);
            let a8 = *a.get_unchecked(i + 7);

            let b1 = *b.get_unchecked(i);
            let b2 = *b.get_unchecked(i + 1);
            let b3 = *b.get_unchecked(i + 2);
            let b4 = *b.get_unchecked(i + 3);
            let b5 = *b.get_unchecked(i + 4);
            let b6 = *b.get_unchecked(i + 5);
            let b7 = *b.get_unchecked(i + 6);
            let b8 = *b.get_unchecked(i + 7);

            let diff1 = <Acc as NumCast>::from(a1).unwrap() - <Acc as NumCast>::from(b1).unwrap();
            let diff2 = <Acc as NumCast>::from(a2).unwrap() - <Acc as NumCast>::from(b2).unwrap();
            let diff3 = <Acc as NumCast>::from(a3).unwrap() - <Acc as NumCast>::from(b3).unwrap();
            let diff4 = <Acc as NumCast>::from(a4).unwrap() - <Acc as NumCast>::from(b4).unwrap();
            let diff5 = <Acc as NumCast>::from(a5).unwrap() - <Acc as NumCast>::from(b5).unwrap();
            let diff6 = <Acc as NumCast>::from(a6).unwrap() - <Acc as NumCast>::from(b6).unwrap();
            let diff7 = <Acc as NumCast>::from(a7).unwrap() - <Acc as NumCast>::from(b7).unwrap();
            let diff8 = <Acc as NumCast>::from(a8).unwrap() - <Acc as NumCast>::from(b8).unwrap();

            acc1 += diff1 * diff1;
            acc2 += diff2 * diff2;
            acc3 += diff3 * diff3;
            acc4 += diff4 * diff4;
            acc5 += diff5 * diff5;
            acc6 += diff6 * diff6;
            acc7 += diff7 * diff7;
            acc8 += diff8 * diff8;
        }

        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        unsafe {
            let a_val = <Acc as NumCast>::from(*a.get_unchecked(i)).unwrap();
            let b_val = <Acc as NumCast>::from(*b.get_unchecked(i)).unwrap();
            let diff = a_val - b_val;
            acc1 += diff * diff;
        }
        i += 1;
    }

    let sum = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8;
    let sum_f32: f32 = NumCast::from(sum).unwrap();

    Some(sum_f32)
}

/// Baseline dot product with 8-way unrolling
pub fn baseline_dot_unrolled<T, Acc>(a: &[T], b: &[T]) -> Option<f32>
where
    T: Num + Copy + NumCast,
    Acc: Num + Copy + NumCast + AddAssign + 'static,
    T: AsPrimitive<Acc>,
{
    if a.len() != b.len() {
        return None;
    }

    let mut i = 0;
    let remainder = a.len() % 8;
    let mut acc1 = Acc::zero();
    let mut acc2 = Acc::zero();
    let mut acc3 = Acc::zero();
    let mut acc4 = Acc::zero();
    let mut acc5 = Acc::zero();
    let mut acc6 = Acc::zero();
    let mut acc7 = Acc::zero();
    let mut acc8 = Acc::zero();

    while i < (a.len() - remainder) {
        unsafe {
            let a1: Acc = NumCast::from(*a.get_unchecked(i)).unwrap();
            let a2: Acc = NumCast::from(*a.get_unchecked(i + 1)).unwrap();
            let a3: Acc = NumCast::from(*a.get_unchecked(i + 2)).unwrap();
            let a4: Acc = NumCast::from(*a.get_unchecked(i + 3)).unwrap();
            let a5: Acc = NumCast::from(*a.get_unchecked(i + 4)).unwrap();
            let a6: Acc = NumCast::from(*a.get_unchecked(i + 5)).unwrap();
            let a7: Acc = NumCast::from(*a.get_unchecked(i + 6)).unwrap();
            let a8: Acc = NumCast::from(*a.get_unchecked(i + 7)).unwrap();

            let b1: Acc = NumCast::from(*b.get_unchecked(i)).unwrap();
            let b2: Acc = NumCast::from(*b.get_unchecked(i + 1)).unwrap();
            let b3: Acc = NumCast::from(*b.get_unchecked(i + 2)).unwrap();
            let b4: Acc = NumCast::from(*b.get_unchecked(i + 3)).unwrap();
            let b5: Acc = NumCast::from(*b.get_unchecked(i + 4)).unwrap();
            let b6: Acc = NumCast::from(*b.get_unchecked(i + 5)).unwrap();
            let b7: Acc = NumCast::from(*b.get_unchecked(i + 6)).unwrap();
            let b8: Acc = NumCast::from(*b.get_unchecked(i + 7)).unwrap();

            acc1 += a1 * b1;
            acc2 += a2 * b2;
            acc3 += a3 * b3;
            acc4 += a4 * b4;
            acc5 += a5 * b5;
            acc6 += a6 * b6;
            acc7 += a7 * b7;
            acc8 += a8 * b8;
        }
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        unsafe {
            let a_acc: Acc = NumCast::from(*a.get_unchecked(i)).unwrap();
            let b_acc: Acc = NumCast::from(*b.get_unchecked(i)).unwrap();
            acc1 += a_acc * b_acc;
        }
        i += 1;
    }

    let sum = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8;
    Some(NumCast::from(sum).unwrap())
}

// endregion

// region: Benchmarks

/// Benchmark squared Euclidean distance
pub fn bench_sqeuclidean(c: &mut Criterion) {
    let dims = get_vector_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("similarity/sqeuclidean");

    // f32
    if should_run_benchmark("similarity/sqeuclidean/f32") {
        let a = generate_random_f32(&mut rng, dims);
        let b = generate_random_f32(&mut rng, dims);
        group.throughput(Throughput::Bytes(
            (dims * std::mem::size_of::<f32>()) as u64,
        ));

        group.bench_function("numkong_f32", |bench| {
            bench.iter(|| black_box(NumKongSpatial::sqeuclidean(&a, &b)))
        });

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_sqeuclidean_unrolled::<f32, f32>(&a, &b)))
        });
    }

    // f64
    if should_run_benchmark("similarity/sqeuclidean/f64") {
        let a = generate_random_f64(&mut rng, dims);
        let b = generate_random_f64(&mut rng, dims);
        group.throughput(Throughput::Bytes(
            (dims * std::mem::size_of::<f64>()) as u64,
        ));

        group.bench_function("numkong_f64", |bench| {
            bench.iter(|| black_box(NumKongSpatial::sqeuclidean(&a, &b)))
        });

        group.bench_function("baseline_f64", |bench| {
            bench.iter(|| black_box(baseline_sqeuclidean_unrolled::<f64, f64>(&a, &b)))
        });
    }

    // i8
    if should_run_benchmark("similarity/sqeuclidean/i8") {
        let a = generate_random_i8(&mut rng, dims);
        let b = generate_random_i8(&mut rng, dims);
        group.throughput(Throughput::Bytes((dims * std::mem::size_of::<i8>()) as u64));

        group.bench_function("numkong_i8", |bench| {
            bench.iter(|| black_box(NumKongSpatial::sqeuclidean(&a, &b)))
        });

        group.bench_function("baseline_i8", |bench| {
            bench.iter(|| black_box(baseline_sqeuclidean_unrolled::<i8, i32>(&a, &b)))
        });
    }

    group.finish();
}

/// Benchmark angular distance
pub fn bench_angular(c: &mut Criterion) {
    let dims = get_vector_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("similarity/angular");

    // f32
    if should_run_benchmark("similarity/angular/f32") {
        let a = generate_random_f32(&mut rng, dims);
        let b = generate_random_f32(&mut rng, dims);
        group.throughput(Throughput::Bytes(
            (dims * std::mem::size_of::<f32>()) as u64,
        ));

        group.bench_function("numkong_f32", |bench| {
            bench.iter(|| black_box(NumKongSpatial::angular(&a, &b)))
        });

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_angular_unrolled::<f32, f32>(&a, &b)))
        });
    }

    // f64
    if should_run_benchmark("similarity/angular/f64") {
        let a = generate_random_f64(&mut rng, dims);
        let b = generate_random_f64(&mut rng, dims);
        group.throughput(Throughput::Bytes(
            (dims * std::mem::size_of::<f64>()) as u64,
        ));

        group.bench_function("numkong_f64", |bench| {
            bench.iter(|| black_box(NumKongSpatial::angular(&a, &b)))
        });

        group.bench_function("baseline_f64", |bench| {
            bench.iter(|| black_box(baseline_angular_unrolled::<f64, f64>(&a, &b)))
        });
    }

    // i8
    if should_run_benchmark("similarity/angular/i8") {
        let a = generate_random_i8(&mut rng, dims);
        let b = generate_random_i8(&mut rng, dims);
        group.throughput(Throughput::Bytes((dims * std::mem::size_of::<i8>()) as u64));

        group.bench_function("numkong_i8", |bench| {
            bench.iter(|| black_box(NumKongSpatial::angular(&a, &b)))
        });

        group.bench_function("baseline_i8", |bench| {
            bench.iter(|| black_box(baseline_angular_unrolled::<i8, i32>(&a, &b)))
        });
    }

    group.finish();
}

/// Benchmark dot product
pub fn bench_dot(c: &mut Criterion) {
    let dims = get_vector_dims();
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("similarity/dot");

    // f32
    if should_run_benchmark("similarity/dot/f32") {
        let a = generate_random_f32(&mut rng, dims);
        let b = generate_random_f32(&mut rng, dims);
        group.throughput(Throughput::Bytes(
            (dims * std::mem::size_of::<f32>()) as u64,
        ));

        group.bench_function("numkong_f32", |bench| {
            bench.iter(|| black_box(NumKongSpatial::dot(&a, &b)))
        });

        group.bench_function("baseline_f32", |bench| {
            bench.iter(|| black_box(baseline_dot_unrolled::<f32, f32>(&a, &b)))
        });
    }

    // f64
    if should_run_benchmark("similarity/dot/f64") {
        let a = generate_random_f64(&mut rng, dims);
        let b = generate_random_f64(&mut rng, dims);
        group.throughput(Throughput::Bytes(
            (dims * std::mem::size_of::<f64>()) as u64,
        ));

        group.bench_function("numkong_f64", |bench| {
            bench.iter(|| black_box(NumKongSpatial::dot(&a, &b)))
        });

        group.bench_function("baseline_f64", |bench| {
            bench.iter(|| black_box(baseline_dot_unrolled::<f64, f64>(&a, &b)))
        });
    }

    group.finish();
}

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_baseline_correctness() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        // Use the unrolled baseline from this module
        let result = baseline_dot_unrolled::<f32, f32>(&a, &b);
        assert!(result.is_some());
        assert!((result.unwrap() - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_sqeuclidean_baseline_correctness() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];

        // Use the unrolled baseline from this module
        let result_sq = baseline_sqeuclidean_unrolled::<f32, f32>(&a, &b);
        assert!(result_sq.is_some());
        let result = result_sq.unwrap().sqrt();
        assert!((result - 5.0).abs() < 1e-6); // sqrt(3^2 + 4^2) = 5
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_dot, bench_angular, bench_sqeuclidean
}
criterion_main!(benches);

// endregion
