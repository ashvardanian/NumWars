//! Benchmark for reduction operations
//!
//! Compares NumKong vs baseline Rust implementations, ndarray, and polars for
//! reduction operations: sum and row-wise norms.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_reduce --bench bench_reduce
//! NUMWARS_FILTER="sum" cargo bench --features bench_reduce
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Tensor size in elements (default: 1000000)
//! - NUMWARS_FILTER: Regex to filter benchmark names
//!
//! Benchmark naming: reduce/{operation}/{dtype}
//! Examples: reduce/sum/f32, reduce/row_norms/f64

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use ndarray::{Array1, Array2};
use num_traits::Float;
use numkong::{bf16, capabilities, f16, Dot, FloatLike, ReduceMoments};
use polars::prelude::*;
use std::hint::black_box;
use std::iter::Sum;
use utils::*;

// region: Operation Model

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
}

impl ReduceOp {
    fn slug(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
        }
    }
}

// endregion

// region: Baseline Kernels

fn baseline_sum<T: Copy + Sum>(data: &[T]) -> T {
    data.iter().copied().sum()
}

fn baseline_row_norms_float<T: Float + Sum>(data: &[T], output: &mut [T], batch_size: usize, ndim: usize) {
    for row in 0..batch_size {
        let row_data = &data[row * ndim..(row + 1) * ndim];
        output[row] = row_data.iter().map(|&v| v * v).sum::<T>().sqrt();
    }
}

// endregion

// region: Per-Library Run Traits

trait RunBaseline: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

impl RunBaseline for f32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32]) {
        match op {
            ReduceOp::Sum => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_sum(data))));
            }
        }
    }
}

impl RunBaseline for f64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64]) {
        match op {
            ReduceOp::Sum => {
                group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_sum(data))));
            }
        }
    }
}

impl RunBaseline for u8 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[u8]) {
        match op {
            ReduceOp::Sum => {
                group.bench_function("baseline", |bench| {
                    bench.iter(|| {
                        let mut acc: u64 = 0;
                        for &v in data {
                            acc += v as u64;
                        }
                        black_box(acc)
                    })
                });
            }
        }
    }
}

impl RunBaseline for bf16 {}

trait RunNumKong: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

impl<T: ReduceMoments> RunNumKong for T {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[Self]) {
        let stride = std::mem::size_of::<T>();
        match op {
            ReduceOp::Sum => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(T::reduce_moments(data, stride).0))
                });
            }
        }
    }
}

trait RunNdarray: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

impl RunNdarray for f32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32]) {
        let array = Array1::from_vec(data.to_vec());
        match op {
            ReduceOp::Sum => {
                group.bench_function("ndarray", |bench| bench.iter(|| black_box(array.sum())));
            }
        }
    }
}

impl RunNdarray for f64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64]) {
        let array = Array1::from_vec(data.to_vec());
        match op {
            ReduceOp::Sum => {
                group.bench_function("ndarray", |bench| bench.iter(|| black_box(array.sum())));
            }
        }
    }
}

impl RunNdarray for u8 {}
impl RunNdarray for bf16 {}

trait RunPolars: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

impl RunPolars for f32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32]) {
        let chunked = Float32Chunked::from_slice("data".into(), data);
        let _ = match op {
            ReduceOp::Sum => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.sum()))),
        };
    }
}

impl RunPolars for f64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64]) {
        let chunked = Float64Chunked::from_slice("data".into(), data);
        let _ = match op {
            ReduceOp::Sum => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.sum()))),
        };
    }
}

impl RunPolars for u8 {}
impl RunPolars for bf16 {}

// endregion

// region: Generic Helpers

fn bench_reduce_op_dtype<T>(c: &mut Criterion, op: ReduceOp, dtype: &str, size: usize, init: T)
where
    T: Clone + RunBaseline + RunNumKong + RunNdarray + RunPolars + 'static,
{
    let name = format!("reduce/{}/{}", op.slug(), dtype);
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((size * std::mem::size_of::<T>()) as u64));

    let data = vec![init; size];
    <T as RunBaseline>::run(op, &mut group, &data);
    <T as RunNumKong>::run(op, &mut group, &data);
    <T as RunNdarray>::run(op, &mut group, &data);
    <T as RunPolars>::run(op, &mut group, &data);

    group.finish();
}

// endregion

// region: Benchmarks

pub fn bench_sum(c: &mut Criterion) {
    capabilities::configure_thread();
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::Sum, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "f64", size, 1.0f64);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "u8", size, 1u8);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "bf16", size, bf16::from_f32(1.0));
}

trait RunRowNorms: Clone + 'static {
    fn bench_all(group: &mut BenchmarkGroup<'_, WallTime>, data: &[Self], batch_size: usize, ndim: usize);
}

trait RunRowNormsNdarray: Clone + 'static {
    fn bench_ndarray(_group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self], _batch_size: usize, _ndim: usize) {}
}

fn bench_row_norms_ndarray<T: Float + Sum + Clone + 'static>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    data: &[T],
    batch_size: usize,
    ndim: usize,
) {
    let matrix = Array2::<T>::from_shape_vec((batch_size, ndim), data.to_vec()).unwrap();
    let mut output = vec![T::zero(); batch_size];
    group.bench_function("ndarray", |bench| {
        bench.iter(|| {
            for (i, row) in matrix.rows().into_iter().enumerate() {
                output[i] = row.dot(&row).sqrt();
            }
            black_box(&output);
        })
    });
}

impl RunRowNormsNdarray for f32 {
    fn bench_ndarray(group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32], batch_size: usize, ndim: usize) {
        bench_row_norms_ndarray(group, data, batch_size, ndim);
    }
}

impl RunRowNormsNdarray for f64 {
    fn bench_ndarray(group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64], batch_size: usize, ndim: usize) {
        bench_row_norms_ndarray(group, data, batch_size, ndim);
    }
}

impl RunRowNormsNdarray for bf16 {}
impl RunRowNormsNdarray for f16 {}

impl RunRowNorms for f32 {
    fn bench_all(group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32], batch_size: usize, ndim: usize) {
        let mut output = vec![0.0f32; batch_size];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_row_norms_float(data, &mut output, batch_size, ndim);
                black_box(&output);
            })
        });
        group.bench_function("numkong", |bench| {
            bench.iter(|| {
                for row in 0..batch_size {
                    let row_data = &data[row * ndim..(row + 1) * ndim];
                    output[row] = (f32::dot(row_data, row_data).unwrap() as f32).sqrt();
                }
                black_box(&output);
            })
        });
        f32::bench_ndarray(group, data, batch_size, ndim);
    }
}

impl RunRowNorms for f64 {
    fn bench_all(group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64], batch_size: usize, ndim: usize) {
        let mut output = vec![0.0f64; batch_size];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_row_norms_float(data, &mut output, batch_size, ndim);
                black_box(&output);
            })
        });
        group.bench_function("numkong", |bench| {
            bench.iter(|| {
                for row in 0..batch_size {
                    let row_data = &data[row * ndim..(row + 1) * ndim];
                    output[row] = f64::dot(row_data, row_data).unwrap().sqrt();
                }
                black_box(&output);
            })
        });
        f64::bench_ndarray(group, data, batch_size, ndim);
    }
}

fn bench_row_norms_mini<T>(group: &mut BenchmarkGroup<'_, WallTime>, data: &[T], batch_size: usize, ndim: usize)
where
    T: FloatLike + Dot + Clone + 'static,
    <T as Dot>::Output: Into<f64>,
{
    let mut output = vec![0.0f32; batch_size];
    group.bench_function("baseline", |bench| {
        bench.iter(|| {
            for row in 0..batch_size {
                let s = &data[row * ndim..(row + 1) * ndim];
                output[row] = s
                    .iter()
                    .map(|v| {
                        let f = v.to_f32();
                        f * f
                    })
                    .sum::<f32>()
                    .sqrt();
            }
            black_box(&output);
        })
    });
    group.bench_function("numkong", |bench| {
        bench.iter(|| {
            for row in 0..batch_size {
                let s = &data[row * ndim..(row + 1) * ndim];
                let dot_f64: f64 = T::dot(s, s).unwrap().into();
                output[row] = (dot_f64 as f32).sqrt();
            }
            black_box(&output);
        })
    });
}

impl RunRowNorms for bf16 {
    fn bench_all(group: &mut BenchmarkGroup<'_, WallTime>, data: &[bf16], batch_size: usize, ndim: usize) {
        bench_row_norms_mini(group, data, batch_size, ndim);
    }
}

impl RunRowNorms for f16 {
    fn bench_all(group: &mut BenchmarkGroup<'_, WallTime>, data: &[f16], batch_size: usize, ndim: usize) {
        bench_row_norms_mini(group, data, batch_size, ndim);
    }
}

fn bench_row_norms_dtype<T: RunRowNorms>(c: &mut Criterion, dtype: &str, batch_size: usize, ndim: usize, init: T) {
    let name = format!("reduce/row_norms/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }

    let total_elements = batch_size * ndim;
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((total_elements * std::mem::size_of::<T>()) as u64));
    let data = vec![init; total_elements];

    T::bench_all(&mut group, &data, batch_size, ndim);
    group.finish();
}

pub fn bench_row_norms(c: &mut Criterion) {
    capabilities::configure_thread();
    let ndim = get_vector_dims();
    let batch_size = get_vector_dims();
    bench_row_norms_dtype(c, "f32", batch_size, ndim, 1.0f32);
    bench_row_norms_dtype(c, "f64", batch_size, ndim, 1.0f64);
    bench_row_norms_dtype(c, "bf16", batch_size, ndim, bf16::from_f32(1.0));
    bench_row_norms_dtype(c, "f16", batch_size, ndim, f16::from_f32(1.0));
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn baseline_sum_correctness() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!((baseline_sum(&data) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn baseline_row_norms_float_correctness() {
        // 2 rows of 2 elements: [3, 4] and [5, 12]
        let data = vec![3.0f32, 4.0, 5.0, 12.0];
        let mut norms = vec![0.0f32; 2];
        baseline_row_norms_float(&data, &mut norms, 2, 2);
        assert!((norms[0] - 5.0).abs() < 1e-6);
        assert!((norms[1] - 13.0).abs() < 1e-6);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_sum, bench_row_norms
}
criterion_main!(benches);

// endregion
