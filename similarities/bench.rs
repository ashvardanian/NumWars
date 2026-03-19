//! Benchmark for many-to-many distance matrix operations
//!
//! Compares NumKong packed matrix operations vs baseline nested-loop implementations
//! for computing N×M distance matrices.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_similarities --bench bench_similarities
//! NUMWARS_FILTER="angulars" cargo bench --features bench_similarities
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Vector dimension (default: 2048)
//! - NUMWARS_BATCH_SIZE: Number of rows in each matrix (default: 1000)
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
use num_traits::{Float, Num, NumCast, Zero};
use numkong::{bf16, capabilities, u1x8, Angulars, Euclideans, PackedMatrix, Tensor};
use std::hint::black_box;
use std::ops::AddAssign;
use utils::*;

// region: Baseline Implementations

trait SpatialType: BaselineConvert<Self::Acc> + Copy + Sized + 'static {
    type Acc: Num + Copy + NumCast + AddAssign + PartialOrd + 'static;
    type Output: Float + NumCast + Copy + 'static;
}

impl SpatialType for f32 {
    type Acc = f32;
    type Output = f32;
}

impl SpatialType for f64 {
    type Acc = f64;
    type Output = f64;
}

impl SpatialType for i8 {
    type Acc = i32;
    type Output = f32;
}

impl SpatialType for u8 {
    type Acc = i32;
    type Output = f32;
}

impl SpatialType for bf16 {
    type Acc = f32;
    type Output = f32;
}

/// Baseline N×M angular distance matrix via nested pairwise computation.
fn baseline_angulars<T: SpatialType>(
    matrix_a: &[T],
    matrix_b: &[T],
    output: &mut [T::Output],
    row_count_a: usize,
    row_count_b: usize,
    dimension: usize,
) {
    let one: T::Output = NumCast::from(1.0).unwrap();

    for row_a in 0..row_count_a {
        for row_b in 0..row_count_b {
            let vector_a = &matrix_a[row_a * dimension..(row_a + 1) * dimension];
            let vector_b = &matrix_b[row_b * dimension..(row_b + 1) * dimension];
            let mut dot_product = T::Acc::zero();
            let mut norm_a_squared = T::Acc::zero();
            let mut norm_b_squared = T::Acc::zero();

            for element in 0..dimension {
                let a: T::Acc = vector_a[element].to_acc();
                let b: T::Acc = vector_b[element].to_acc();
                dot_product += a * b;
                norm_a_squared += a * a;
                norm_b_squared += b * b;
            }

            let dot_out: T::Output = NumCast::from(dot_product).unwrap();
            let norm_a_out: T::Output = NumCast::from(norm_a_squared).unwrap();
            let norm_b_out: T::Output = NumCast::from(norm_b_squared).unwrap();
            output[row_a * row_count_b + row_b] = one - dot_out / (norm_a_out.sqrt() * norm_b_out.sqrt());
        }
    }
}

/// Baseline N×M Euclidean distance matrix.
fn baseline_euclideans<T: SpatialType>(
    matrix_a: &[T],
    matrix_b: &[T],
    output: &mut [T::Output],
    row_count_a: usize,
    row_count_b: usize,
    dimension: usize,
) {
    for row_a in 0..row_count_a {
        for row_b in 0..row_count_b {
            let vector_a = &matrix_a[row_a * dimension..(row_a + 1) * dimension];
            let vector_b = &matrix_b[row_b * dimension..(row_b + 1) * dimension];
            let mut sum_squared_diff = T::Acc::zero();

            for element in 0..dimension {
                let a: T::Acc = vector_a[element].to_acc();
                let b: T::Acc = vector_b[element].to_acc();
                let diff = a - b;
                sum_squared_diff += diff * diff;
            }

            let sum_out: T::Output = NumCast::from(sum_squared_diff).unwrap();
            output[row_a * row_count_b + row_b] = sum_out.sqrt();
        }
    }
}

/// Baseline N×M Hamming distance matrix for bit-packed vectors.
fn baseline_hammings_u1x8(
    matrix_a: &[u1x8],
    matrix_b: &[u1x8],
    output: &mut [u32],
    row_count_a: usize,
    row_count_b: usize,
    byte_count: usize,
) {
    for row_a in 0..row_count_a {
        for row_b in 0..row_count_b {
            let vector_a = &matrix_a[row_a * byte_count..(row_a + 1) * byte_count];
            let vector_b = &matrix_b[row_b * byte_count..(row_b + 1) * byte_count];
            let distance: u32 = vector_a
                .iter()
                .zip(vector_b)
                .map(|(a, b)| (a.bits() ^ b.bits()).count_ones())
                .sum();
            output[row_a * row_count_b + row_b] = distance;
        }
    }
}

/// Baseline N×M Jaccard distance matrix for bit-packed vectors.
fn baseline_jaccards_u1x8(
    matrix_a: &[u1x8],
    matrix_b: &[u1x8],
    output: &mut [f32],
    row_count_a: usize,
    row_count_b: usize,
    byte_count: usize,
) {
    for row_a in 0..row_count_a {
        for row_b in 0..row_count_b {
            let vector_a = &matrix_a[row_a * byte_count..(row_a + 1) * byte_count];
            let vector_b = &matrix_b[row_b * byte_count..(row_b + 1) * byte_count];
            let (intersection_count, union_count) =
                vector_a
                    .iter()
                    .zip(vector_b)
                    .fold((0u32, 0u32), |(inter, uni), (a, b)| {
                        let bits_a = a.bits();
                        let bits_b = b.bits();
                        (
                            inter + (bits_a & bits_b).count_ones(),
                            uni + (bits_a | bits_b).count_ones(),
                        )
                    });
            output[row_a * row_count_b + row_b] = if union_count == 0 {
                0.0
            } else {
                1.0 - (intersection_count as f32 / union_count as f32)
            };
        }
    }
}

// endregion

// region: Per-library Run traits

trait RunBaselineAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl<T: SpatialType> RunBaselineAngulars for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], data_b: &[T], bs: usize, dim: usize) {
        let mut output = vec![T::Output::zero(); bs * bs];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_angulars(data_a, data_b, &mut output, bs, bs, dim);
                black_box(&output);
            })
        });
    }
}

trait RunNumKongAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl<T: Angulars + Clone + 'static> RunNumKongAngulars for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], data_b: &[T], bs: usize, dim: usize) {
        let tensor_a = Tensor::<T>::try_from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let tensor_b = Tensor::<T>::try_from_slice(data_b, &[bs, dim]).expect("Failed to create tensor B");
        let packed_b = PackedMatrix::try_pack(&tensor_b).expect("Failed to pack B");
        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(tensor_a.angulars_packed(&packed_b)))
        });
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

trait RunBaselineEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl<T: SpatialType> RunBaselineEuclideans for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], data_b: &[T], bs: usize, dim: usize) {
        let mut output = vec![T::Output::zero(); bs * bs];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_euclideans(data_a, data_b, &mut output, bs, bs, dim);
                black_box(&output);
            })
        });
    }
}

trait RunNumKongEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
}

impl<T: Euclideans + Clone + 'static> RunNumKongEuclideans for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], data_b: &[T], bs: usize, dim: usize) {
        let tensor_a = Tensor::<T>::try_from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let tensor_b = Tensor::<T>::try_from_slice(data_b, &[bs, dim]).expect("Failed to create tensor B");
        let packed_b = PackedMatrix::try_pack(&tensor_b).expect("Failed to pack B");
        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(tensor_a.euclideans_packed(&packed_b)))
        });
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
    T: Clone + RunBaselineAngulars + RunNumKongAngulars + RunNdarrayAngulars + RunNalgebraAngulars + 'static,
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
    <T as RunBaselineAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    <T as RunNdarrayAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    <T as RunNalgebraAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
    group.finish();
}

fn bench_euclideans_dtype<T>(c: &mut Criterion, dtype: &str, batch_size: usize, dimension: usize, init: T)
where
    T: Clone + RunBaselineEuclideans + RunNumKongEuclideans + RunNdarrayEuclideans + RunNalgebraEuclideans + 'static,
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
    <T as RunBaselineEuclideans>::run(&mut group, &data_a, &data_b, batch_size, dimension);
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

        let mut output = vec![0u32; batch_size * batch_size];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_hammings_u1x8(
                    &matrix_a_data,
                    &matrix_b_data,
                    &mut output,
                    batch_size,
                    batch_size,
                    byte_count,
                );
                black_box(&output);
            })
        });

        let tensor_a = Tensor::<u1x8>::try_from_slice(&matrix_a_data, &[batch_size, byte_count])
            .expect("Failed to create tensor A");
        let tensor_b = Tensor::<u1x8>::try_from_slice(&matrix_b_data, &[batch_size, byte_count])
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

        let mut output = vec![0.0f32; batch_size * batch_size];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_jaccards_u1x8(
                    &matrix_a_data,
                    &matrix_b_data,
                    &mut output,
                    batch_size,
                    batch_size,
                    byte_count,
                );
                black_box(&output);
            })
        });

        let tensor_a = Tensor::<u1x8>::try_from_slice(&matrix_a_data, &[batch_size, byte_count])
            .expect("Failed to create tensor A");
        let tensor_b = Tensor::<u1x8>::try_from_slice(&matrix_b_data, &[batch_size, byte_count])
            .expect("Failed to create tensor B");
        let packed_b = PackedMatrix::try_pack(&tensor_b).expect("Failed to pack B");

        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(tensor_a.jaccards_packed(&packed_b)))
        });
        group.finish();
    }
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_angulars_identical_vectors() {
        let dimension = 4;
        let matrix_a = vec![1.0f32, 0.0, 0.0, 0.0];
        let matrix_b = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut output = vec![0.0f32; 1];
        baseline_angulars(&matrix_a, &matrix_b, &mut output, 1, 1, dimension);
        assert!(output[0].abs() < 1e-6); // identical vectors = 0 angular distance
    }

    #[test]
    fn baseline_euclideans_known_distance() {
        let dimension = 2;
        let matrix_a = vec![0.0f32, 0.0];
        let matrix_b = vec![3.0f32, 4.0];
        let mut output = vec![0.0f32; 1];
        baseline_euclideans(&matrix_a, &matrix_b, &mut output, 1, 1, dimension);
        assert!((output[0] - 5.0).abs() < 1e-6); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn baseline_angulars_i8_output_type() {
        let dimension = 2;
        let matrix_a = vec![1i8, 0];
        let matrix_b = vec![0i8, 1];
        let mut output = vec![0.0f32; 1];
        baseline_angulars(&matrix_a, &matrix_b, &mut output, 1, 1, dimension);
        assert!((output[0] - 1.0).abs() < 1e-6);
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
