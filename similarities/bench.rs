//! Benchmark for many-to-many distance matrix operations
//!
//! Compares NumKong packed matrix operations vs BLAS-backed libraries
//! for computing N×M distance matrices.
//!
//! Competitors:
//! - numkong (PackedMatrix SIMD)
//! - ndarray (BLAS-backed matmul for f32/f64)
//! - nalgebra (matmul-based for f32/f64)
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_similarities --bench bench_similarities
//! NUMWARS_FILTER="angulars" cargo bench --features bench_similarities
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Vector dimension and row count (default: 2048)
//! - NUMWARS_FILTER: Regex to filter benchmark names
//!
//! Benchmark naming: similarities/{metric}/{dtype}
//! Examples: similarities/angulars/f32, similarities/euclideans/f64

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use nalgebra::DMatrix;
use ndarray::Array2;
use numkong::{bf16, capabilities, u1x8, Angulars, Euclideans, PackedMatrix, Tensor};
use std::hint::black_box;
use utils::*;

// region: Per-library Run traits

trait RunNumKongAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl<T: Angulars + Clone + Send + Sync + 'static> RunNumKongAngulars for T
where
    T::Accumulator: Send + Sync,
    <T as Angulars>::SpatialResult: Send + Sync,
{
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], data_b: &[T], bs: usize, dim: usize) {
        let threads = get_thread_count();
        let tensor_a = Tensor::<T>::try_from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let tensor_b = Tensor::<T>::try_from_slice(data_b, &[bs, dim]).expect("Failed to create tensor B");
        let packed_b = PackedMatrix::try_pack(&tensor_b).expect("Failed to pack B");
        if threads > 1 {
            let mut pool = fork_union::ThreadPool::try_spawn(threads).expect("Failed to spawn thread pool");
            pool.for_threads(|_, _| { capabilities::configure_thread(); }).join();
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(tensor_a.angulars_packed_parallel(&packed_b, &mut pool)))
            });
        } else {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(tensor_a.angulars_packed(&packed_b)))
            });
        }
    }
}

trait RunNdarrayAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNdarrayAngulars for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = Array2::<f32>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f32>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a: Vec<f32> = matrix_a.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let norms_b: Vec<f32> = matrix_b.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let mut output = Array2::<f32>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayAngulars for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = Array2::<f64>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f64>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a: Vec<f64> = matrix_a.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let norms_b: Vec<f64> = matrix_b.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let mut output = Array2::<f64>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayAngulars for i8 {}
impl RunNdarrayAngulars for u8 {}
impl RunNdarrayAngulars for bf16 {}

trait RunNalgebraAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNalgebraAngulars for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f32>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f32>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a: Vec<f32> = (0..bs).map(|i| matrix_a.row(i).norm()).collect();
        let norms_b: Vec<f32> = (0..bs).map(|i| matrix_b.row(i).norm()).collect();
        let mut output = DMatrix::<f32>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraAngulars for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f64>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f64>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a: Vec<f64> = (0..bs).map(|i| matrix_a.row(i).norm()).collect();
        let norms_b: Vec<f64> = (0..bs).map(|i| matrix_b.row(i).norm()).collect();
        let mut output = DMatrix::<f64>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraAngulars for i8 {}
impl RunNalgebraAngulars for u8 {}
impl RunNalgebraAngulars for bf16 {}

trait RunNumKongEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl<T: Euclideans + Clone + Send + Sync + 'static> RunNumKongEuclideans for T
where
    T::Accumulator: Send + Sync,
    <T as Euclideans>::SpatialResult: Send + Sync,
{
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], data_b: &[T], bs: usize, dim: usize) {
        let threads = get_thread_count();
        let tensor_a = Tensor::<T>::try_from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let tensor_b = Tensor::<T>::try_from_slice(data_b, &[bs, dim]).expect("Failed to create tensor B");
        let packed_b = PackedMatrix::try_pack(&tensor_b).expect("Failed to pack B");
        if threads > 1 {
            let mut pool = fork_union::ThreadPool::try_spawn(threads).expect("Failed to spawn thread pool");
            pool.for_threads(|_, _| { capabilities::configure_thread(); }).join();
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(tensor_a.euclideans_packed_parallel(&packed_b, &mut pool)))
            });
        } else {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(tensor_a.euclideans_packed(&packed_b)))
            });
        }
    }
}

trait RunNdarrayEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNdarrayEuclideans for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = Array2::<f32>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f32>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a_sq: Vec<f32> = matrix_a.rows().into_iter().map(|row| row.dot(&row)).collect();
        let norms_b_sq: Vec<f32> = matrix_b.rows().into_iter().map(|row| row.dot(&row)).collect();
        let mut output = Array2::<f32>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayEuclideans for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = Array2::<f64>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f64>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a_sq: Vec<f64> = matrix_a.rows().into_iter().map(|row| row.dot(&row)).collect();
        let norms_b_sq: Vec<f64> = matrix_b.rows().into_iter().map(|row| row.dot(&row)).collect();
        let mut output = Array2::<f64>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayEuclideans for i8 {}
impl RunNdarrayEuclideans for u8 {}
impl RunNdarrayEuclideans for bf16 {}

trait RunNalgebraEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNalgebraEuclideans for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f32>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f32>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a_sq: Vec<f32> = (0..bs).map(|i| matrix_a.row(i).dot(&matrix_a.row(i))).collect();
        let norms_b_sq: Vec<f32> = (0..bs).map(|i| matrix_b.row(i).dot(&matrix_b.row(i))).collect();
        let mut output = DMatrix::<f32>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraEuclideans for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f64>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f64>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a_sq: Vec<f64> = (0..bs).map(|i| matrix_a.row(i).dot(&matrix_a.row(i))).collect();
        let norms_b_sq: Vec<f64> = (0..bs).map(|i| matrix_b.row(i).dot(&matrix_b.row(i))).collect();
        let mut output = DMatrix::<f64>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraEuclideans for i8 {}
impl RunNalgebraEuclideans for u8 {}
impl RunNalgebraEuclideans for bf16 {}

// endregion

// region: Generic helpers

fn bench_angulars_dtype<T>(c: &mut Criterion, dtype: &str, batch_size: usize, dimension: usize, init: T)
where
    T: Clone + RunNumKongAngulars + RunNdarrayAngulars + RunNalgebraAngulars + 'static,
{
    let name = format!("similarities/angulars/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    let total_elements = batch_size * dimension;
    let data_a = vec![init.clone(); total_elements];
    let data_b = vec![init; total_elements];
    group.throughput(Throughput::Bytes(
        (2 * total_elements * std::mem::size_of::<T>()) as u64,
    ));

    <T as RunNumKongAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    <T as RunNdarrayAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    <T as RunNalgebraAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    group.finish();
}

fn bench_euclideans_dtype<T>(c: &mut Criterion, dtype: &str, batch_size: usize, dimension: usize, init: T)
where
    T: Clone + RunNumKongEuclideans + RunNdarrayEuclideans + RunNalgebraEuclideans + 'static,
{
    let name = format!("similarities/euclideans/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    let total_elements = batch_size * dimension;
    let data_a = vec![init.clone(); total_elements];
    let data_b = vec![init; total_elements];
    group.throughput(Throughput::Bytes(
        (2 * total_elements * std::mem::size_of::<T>()) as u64,
    ));

    <T as RunNumKongEuclideans>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    <T as RunNdarrayEuclideans>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    <T as RunNalgebraEuclideans>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    group.finish();
}

// endregion

// region: Benchmarks

/// Benchmark N×M angular distance matrix.
pub fn bench_angulars(c: &mut Criterion) {
    capabilities::configure_thread();
    let dimension = get_vector_dims();
    let batch_size = get_batch_size();
    bench_angulars_dtype(c, "f32", batch_size, dimension, 1.0f32);
    bench_angulars_dtype(c, "f64", batch_size, dimension, 1.0f64);
    bench_angulars_dtype(c, "i8", batch_size, dimension, 1i8);
    bench_angulars_dtype(c, "u8", batch_size, dimension, 1u8);
    bench_angulars_dtype(c, "bf16", batch_size, dimension, bf16::from_f32(1.0));
}

/// Benchmark N×M Euclidean distance matrix.
pub fn bench_euclideans(c: &mut Criterion) {
    capabilities::configure_thread();
    let dimension = get_vector_dims();
    let batch_size = get_batch_size();
    bench_euclideans_dtype(c, "f32", batch_size, dimension, 1.0f32);
    bench_euclideans_dtype(c, "f64", batch_size, dimension, 1.0f64);
    bench_euclideans_dtype(c, "i8", batch_size, dimension, 1i8);
    bench_euclideans_dtype(c, "u8", batch_size, dimension, 1u8);
    bench_euclideans_dtype(c, "bf16", batch_size, dimension, bf16::from_f32(1.0));
}

/// Benchmark N×M Hamming distance matrix.
pub fn bench_hammings(c: &mut Criterion) {
    capabilities::configure_thread();
    let dimension = get_vector_dims();
    let byte_count = dimension.div_ceil(8);
    let batch_size = get_batch_size();

    if should_run_benchmark("similarities/hammings/u1x8") {
        let mut group = c.benchmark_group("similarities/hammings/u1x8");
        let total_bytes = batch_size * byte_count;
        let matrix_a_data = vec![u1x8::new(0xAA); total_bytes];
        let matrix_b_data = vec![u1x8::new(0x55); total_bytes];
        group.throughput(Throughput::Bytes((total_bytes * 2) as u64));

        let tensor_a = Tensor::<u1x8>::try_from_slice(&matrix_a_data, &[batch_size, dimension])
            .expect("Failed to create tensor A");
        let tensor_b = Tensor::<u1x8>::try_from_slice(&matrix_b_data, &[batch_size, dimension])
            .expect("Failed to create tensor B");
        let packed_b = PackedMatrix::try_pack(&tensor_b).expect("Failed to pack B");

        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(tensor_a.hammings_packed(&packed_b)))
        });
        group.finish();
    }
}

/// Benchmark N×M Jaccard distance matrix.
pub fn bench_jaccards(c: &mut Criterion) {
    capabilities::configure_thread();
    let dimension = get_vector_dims();
    let byte_count = dimension.div_ceil(8);
    let batch_size = get_batch_size();

    if should_run_benchmark("similarities/jaccards/u1x8") {
        let mut group = c.benchmark_group("similarities/jaccards/u1x8");
        let total_bytes = batch_size * byte_count;
        let matrix_a_data = vec![u1x8::new(0xAA); total_bytes];
        let matrix_b_data = vec![u1x8::new(0x55); total_bytes];
        group.throughput(Throughput::Bytes((total_bytes * 2) as u64));

        let tensor_a = Tensor::<u1x8>::try_from_slice(&matrix_a_data, &[batch_size, dimension])
            .expect("Failed to create tensor A");
        let tensor_b = Tensor::<u1x8>::try_from_slice(&matrix_b_data, &[batch_size, dimension])
            .expect("Failed to create tensor B");
        let packed_b = PackedMatrix::try_pack(&tensor_b).expect("Failed to pack B");

        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(tensor_a.jaccards_packed(&packed_b)))
        });
        group.finish();
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_angulars, bench_euclideans, bench_hammings, bench_jaccards
}
criterion_main!(benches);

// endregion
