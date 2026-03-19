//! Benchmark for reduction operations
//!
//! Compares NumKong vs baseline Rust implementations, ndarray, polars, and argminmax for
//! reduction operations like sum, norm, min, max, argmin, argmax, moments, and minmax.
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
//! Examples: reduce/sum/f32, reduce/moments/f64, reduce/minmax/f32

#[path = "../utils.rs"]
mod utils;

use argminmax::ArgMinMax;
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use ndarray::{Array1, Array2};
use num_traits::Float;
use numkong::{bf16, capabilities, f16, Dot, FloatLike, ReduceMinMax, ReduceMoments};
use polars::prelude::*;
use std::hint::black_box;
use std::iter::Sum;
use utils::*;

// region: Operation model

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Norm,
    Min,
    Max,
    ArgMin,
    ArgMax,
    Moments,
    MinMax,
    RowNorms,
}

impl ReduceOp {
    fn slug(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Norm => "norm",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
            ReduceOp::ArgMin => "argmin",
            ReduceOp::ArgMax => "argmax",
            ReduceOp::Moments => "moments",
            ReduceOp::MinMax => "minmax",
            ReduceOp::RowNorms => "row_norms",
        }
    }
}

// endregion

// region: Baseline kernels

fn baseline_sum<T: Copy + Sum>(data: &[T]) -> T {
    data.iter().copied().sum()
}

fn baseline_norm<T: Float + Sum>(data: &[T]) -> T {
    data.iter().map(|&value| value * value).sum::<T>().sqrt()
}

fn baseline_moments<T: Float + Sum>(data: &[T]) -> (T, T) {
    let count = T::from(data.len()).unwrap();
    let mean = data.iter().copied().sum::<T>() / count;
    let variance = data.iter().map(|&value| (value - mean).powi(2)).sum::<T>() / count;
    (mean, variance)
}

fn baseline_min_float<T: Float>(data: &[T]) -> T {
    data.iter()
        .fold(T::infinity(), |accumulator, &value| accumulator.min(value))
}

fn baseline_max_float<T: Float>(data: &[T]) -> T {
    data.iter()
        .fold(T::neg_infinity(), |accumulator, &value| accumulator.max(value))
}

fn baseline_argmin_float<T: Float>(data: &[T]) -> usize {
    data.iter()
        .enumerate()
        .fold((0, T::infinity()), |(min_index, min_value), (index, &value)| {
            if value < min_value {
                (index, value)
            } else {
                (min_index, min_value)
            }
        })
        .0
}

fn baseline_argmax_float<T: Float>(data: &[T]) -> usize {
    data.iter()
        .enumerate()
        .fold((0, T::neg_infinity()), |(max_index, max_value), (index, &value)| {
            if value > max_value {
                (index, value)
            } else {
                (max_index, max_value)
            }
        })
        .0
}

fn baseline_minmax_float<T: Float>(data: &[T]) -> (T, usize, T, usize) {
    let mut min_value = T::infinity();
    let mut min_index = 0;
    let mut max_value = T::neg_infinity();
    let mut max_index = 0;
    for (index, &value) in data.iter().enumerate() {
        if value < min_value {
            min_value = value;
            min_index = index;
        }
        if value > max_value {
            max_value = value;
            max_index = index;
        }
    }
    (min_value, min_index, max_value, max_index)
}

fn baseline_sum_int<T: Copy + Into<i64>>(data: &[T]) -> i64 {
    let mut accumulator: i64 = 0;
    for &value in data {
        accumulator += value.into();
    }
    accumulator
}

fn baseline_min_ord<T: Copy + PartialOrd>(data: &[T]) -> T {
    let mut min_value = data[0];
    for &value in &data[1..] {
        if value < min_value {
            min_value = value;
        }
    }
    min_value
}

fn baseline_max_ord<T: Copy + PartialOrd>(data: &[T]) -> T {
    let mut max_value = data[0];
    for &value in &data[1..] {
        if value > max_value {
            max_value = value;
        }
    }
    max_value
}

fn baseline_argmin_ord<T: Copy + PartialOrd>(data: &[T]) -> usize {
    let mut min_index = 0;
    let mut min_value = data[0];
    for (index, &value) in data.iter().enumerate().skip(1) {
        if value < min_value {
            min_value = value;
            min_index = index;
        }
    }
    min_index
}

fn baseline_argmax_ord<T: Copy + PartialOrd>(data: &[T]) -> usize {
    let mut max_index = 0;
    let mut max_value = data[0];
    for (index, &value) in data.iter().enumerate().skip(1) {
        if value > max_value {
            max_value = value;
            max_index = index;
        }
    }
    max_index
}

fn baseline_minmax_ord<T: Copy + PartialOrd>(data: &[T]) -> (T, usize, T, usize) {
    let mut min_value = data[0];
    let mut min_index = 0;
    let mut max_value = data[0];
    let mut max_index = 0;
    for (index, &value) in data.iter().enumerate().skip(1) {
        if value < min_value {
            min_value = value;
            min_index = index;
        }
        if value > max_value {
            max_value = value;
            max_index = index;
        }
    }
    (min_value, min_index, max_value, max_index)
}

fn baseline_row_norms_float<T: Float + Sum>(data: &[T], output: &mut [T], batch_size: usize, ndim: usize) {
    for row in 0..batch_size {
        let row_data = &data[row * ndim..(row + 1) * ndim];
        output[row] = row_data.iter().map(|&v| v * v).sum::<T>().sqrt();
    }
}

// endregion

// region: Backend run traits

trait RunBaseline: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

fn run_baseline_float<T: Float + Sum>(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[T]) {
    match op {
        ReduceOp::Sum => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_sum(data))));
        }
        ReduceOp::Norm => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_norm(data))));
        }
        ReduceOp::Min => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_min_float(data))));
        }
        ReduceOp::Max => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_max_float(data))));
        }
        ReduceOp::ArgMin => {
            group.bench_function("baseline", |bench| {
                bench.iter(|| black_box(baseline_argmin_float(data)))
            });
        }
        ReduceOp::ArgMax => {
            group.bench_function("baseline", |bench| {
                bench.iter(|| black_box(baseline_argmax_float(data)))
            });
        }
        ReduceOp::Moments => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_moments(data))));
        }
        ReduceOp::MinMax => {
            group.bench_function("baseline", |bench| {
                bench.iter(|| black_box(baseline_minmax_float(data)))
            });
        }
        ReduceOp::RowNorms => {} // handled directly in bench_row_norms
    }
}

fn run_baseline_int<T: Copy + PartialOrd + Into<i64>>(
    op: ReduceOp,
    group: &mut BenchmarkGroup<'_, WallTime>,
    data: &[T],
) {
    match op {
        ReduceOp::Sum => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_sum_int(data))));
        }
        ReduceOp::Min => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_min_ord(data))));
        }
        ReduceOp::Max => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_max_ord(data))));
        }
        ReduceOp::ArgMin => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_argmin_ord(data))));
        }
        ReduceOp::ArgMax => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_argmax_ord(data))));
        }
        ReduceOp::MinMax => {
            group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_minmax_ord(data))));
        }
        ReduceOp::Norm | ReduceOp::Moments | ReduceOp::RowNorms => {}
    }
}

impl RunBaseline for f32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32]) {
        run_baseline_float(op, group, data);
    }
}

impl RunBaseline for f64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64]) {
        run_baseline_float(op, group, data);
    }
}

impl RunBaseline for i8 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i8]) {
        run_baseline_int(op, group, data);
    }
}

impl RunBaseline for i16 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i16]) {
        run_baseline_int(op, group, data);
    }
}

impl RunBaseline for i32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i32]) {
        run_baseline_int(op, group, data);
    }
}

impl RunBaseline for i64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i64]) {
        run_baseline_int(op, group, data);
    }
}

trait RunNumKong: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

fn run_numkong_reduce<T: ReduceMoments + ReduceMinMax>(
    op: ReduceOp,
    group: &mut BenchmarkGroup<'_, WallTime>,
    data: &[T],
) {
    let stride = std::mem::size_of::<T>();
    match op {
        ReduceOp::Sum => {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(T::reduce_moments(data, stride).0))
            });
        }
        ReduceOp::Min => {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(T::reduce_minmax(data, stride).map(|result| result.0)))
            });
        }
        ReduceOp::Max => {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(T::reduce_minmax(data, stride).map(|result| result.2)))
            });
        }
        ReduceOp::ArgMin => {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(T::reduce_minmax(data, stride).map(|result| result.1)))
            });
        }
        ReduceOp::ArgMax => {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(T::reduce_minmax(data, stride).map(|result| result.3)))
            });
        }
        ReduceOp::Moments => {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(T::reduce_moments(data, stride)))
            });
        }
        ReduceOp::MinMax => {
            group.bench_function("numkong", |bench| {
                bench.iter(|| black_box(T::reduce_minmax(data, stride)))
            });
        }
        ReduceOp::Norm | ReduceOp::RowNorms => {} // handled directly in bench_row_norms
    }
}

impl<T: ReduceMoments + ReduceMinMax> RunNumKong for T {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[Self]) {
        run_numkong_reduce(op, group, data);
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
            ReduceOp::Norm => {
                group.bench_function("ndarray", |bench| bench.iter(|| black_box(array.dot(&array).sqrt())));
            }
            ReduceOp::Min
            | ReduceOp::Max
            | ReduceOp::ArgMin
            | ReduceOp::ArgMax
            | ReduceOp::Moments
            | ReduceOp::MinMax
            | ReduceOp::RowNorms => {} // RowNorms handled directly in bench_row_norms
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
            ReduceOp::Norm => {
                group.bench_function("ndarray", |bench| bench.iter(|| black_box(array.dot(&array).sqrt())));
            }
            ReduceOp::Min
            | ReduceOp::Max
            | ReduceOp::ArgMin
            | ReduceOp::ArgMax
            | ReduceOp::Moments
            | ReduceOp::MinMax
            | ReduceOp::RowNorms => {} // RowNorms handled directly in bench_row_norms
        }
    }
}

impl RunNdarray for i8 {}
impl RunNdarray for i16 {}
impl RunNdarray for i32 {}
impl RunNdarray for i64 {}

trait RunPolars: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

impl RunPolars for f32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32]) {
        let chunked = Float32Chunked::from_slice("data".into(), data);
        let _ = match op {
            ReduceOp::Sum => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.sum()))),
            ReduceOp::Min => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.min()))),
            ReduceOp::Max => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.max()))),
            ReduceOp::Norm
            | ReduceOp::ArgMin
            | ReduceOp::ArgMax
            | ReduceOp::Moments
            | ReduceOp::MinMax
            | ReduceOp::RowNorms => group,
        };
    }
}

impl RunPolars for f64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64]) {
        let chunked = Float64Chunked::from_slice("data".into(), data);
        let _ = match op {
            ReduceOp::Sum => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.sum()))),
            ReduceOp::Min => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.min()))),
            ReduceOp::Max => group.bench_function("polars", |bench| bench.iter(|| black_box(chunked.max()))),
            ReduceOp::Norm
            | ReduceOp::ArgMin
            | ReduceOp::ArgMax
            | ReduceOp::Moments
            | ReduceOp::MinMax
            | ReduceOp::RowNorms => group,
        };
    }
}

impl RunPolars for i8 {}
impl RunPolars for i16 {}
impl RunPolars for i32 {}
impl RunPolars for i64 {}

trait RunArgminmax: Sized {
    fn run(_op: ReduceOp, _group: &mut BenchmarkGroup<'_, WallTime>, _data: &[Self]) {}
}

fn run_argminmax_backend<T>(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[T])
where
    for<'a> &'a [T]: ArgMinMax,
{
    match op {
        ReduceOp::ArgMin => {
            group.bench_function("argminmax", |bench| bench.iter(|| black_box(data.argmin())));
        }
        ReduceOp::ArgMax => {
            group.bench_function("argminmax", |bench| bench.iter(|| black_box(data.argmax())));
        }
        ReduceOp::Sum
        | ReduceOp::Norm
        | ReduceOp::Min
        | ReduceOp::Max
        | ReduceOp::Moments
        | ReduceOp::MinMax
        | ReduceOp::RowNorms => {}
    }
}

impl RunArgminmax for f32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f32]) {
        run_argminmax_backend(op, group, data);
    }
}

impl RunArgminmax for f64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[f64]) {
        run_argminmax_backend(op, group, data);
    }
}

impl RunArgminmax for i8 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i8]) {
        run_argminmax_backend(op, group, data);
    }
}

impl RunArgminmax for i16 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i16]) {
        run_argminmax_backend(op, group, data);
    }
}

impl RunArgminmax for i32 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i32]) {
        run_argminmax_backend(op, group, data);
    }
}

impl RunArgminmax for i64 {
    fn run(op: ReduceOp, group: &mut BenchmarkGroup<'_, WallTime>, data: &[i64]) {
        run_argminmax_backend(op, group, data);
    }
}

// endregion

// region: Generic helper

fn bench_reduce_op_dtype<T>(c: &mut Criterion, op: ReduceOp, dtype: &str, size: usize, init: T)
where
    T: Clone + RunBaseline + RunNumKong + RunNdarray + RunPolars + RunArgminmax + 'static,
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
    <T as RunArgminmax>::run(op, &mut group, &data);

    group.finish();
}

// endregion

// region: Benchmarks

pub fn bench_sum(c: &mut Criterion) {
    capabilities::configure_thread();
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::Sum, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "f64", size, 1.0f64);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "i8", size, 1i8);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "i16", size, 1i16);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "i32", size, 1i32);
    bench_reduce_op_dtype(c, ReduceOp::Sum, "i64", size, 1i64);
}

pub fn bench_norm(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::Norm, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::Norm, "f64", size, 1.0f64);
}

pub fn bench_min(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::Min, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::Min, "f64", size, 1.0f64);
    bench_reduce_op_dtype(c, ReduceOp::Min, "i8", size, 1i8);
    bench_reduce_op_dtype(c, ReduceOp::Min, "i16", size, 1i16);
    bench_reduce_op_dtype(c, ReduceOp::Min, "i32", size, 1i32);
    bench_reduce_op_dtype(c, ReduceOp::Min, "i64", size, 1i64);
}

pub fn bench_max(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::Max, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::Max, "f64", size, 1.0f64);
    bench_reduce_op_dtype(c, ReduceOp::Max, "i8", size, 1i8);
    bench_reduce_op_dtype(c, ReduceOp::Max, "i16", size, 1i16);
    bench_reduce_op_dtype(c, ReduceOp::Max, "i32", size, 1i32);
    bench_reduce_op_dtype(c, ReduceOp::Max, "i64", size, 1i64);
}

pub fn bench_argmin(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::ArgMin, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::ArgMin, "f64", size, 1.0f64);
    bench_reduce_op_dtype(c, ReduceOp::ArgMin, "i8", size, 1i8);
    bench_reduce_op_dtype(c, ReduceOp::ArgMin, "i16", size, 1i16);
    bench_reduce_op_dtype(c, ReduceOp::ArgMin, "i32", size, 1i32);
    bench_reduce_op_dtype(c, ReduceOp::ArgMin, "i64", size, 1i64);
}

pub fn bench_argmax(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::ArgMax, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::ArgMax, "f64", size, 1.0f64);
    bench_reduce_op_dtype(c, ReduceOp::ArgMax, "i8", size, 1i8);
    bench_reduce_op_dtype(c, ReduceOp::ArgMax, "i16", size, 1i16);
    bench_reduce_op_dtype(c, ReduceOp::ArgMax, "i32", size, 1i32);
    bench_reduce_op_dtype(c, ReduceOp::ArgMax, "i64", size, 1i64);
}

pub fn bench_moments(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::Moments, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::Moments, "f64", size, 1.0f64);
}

pub fn bench_minmax(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_reduce_op_dtype(c, ReduceOp::MinMax, "f32", size, 1.0f32);
    bench_reduce_op_dtype(c, ReduceOp::MinMax, "f64", size, 1.0f64);
    bench_reduce_op_dtype(c, ReduceOp::MinMax, "i8", size, 1i8);
    bench_reduce_op_dtype(c, ReduceOp::MinMax, "i16", size, 1i16);
    bench_reduce_op_dtype(c, ReduceOp::MinMax, "i32", size, 1i32);
    bench_reduce_op_dtype(c, ReduceOp::MinMax, "i64", size, 1i64);
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
    let batch_size = get_batch_size();
    bench_row_norms_dtype(c, "f32", batch_size, ndim, 1.0f32);
    bench_row_norms_dtype(c, "f64", batch_size, ndim, 1.0f64);
    bench_row_norms_dtype(c, "bf16", batch_size, ndim, bf16::from_f32(1.0));
    bench_row_norms_dtype(c, "f16", batch_size, ndim, f16::from_f32(1.0));
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_sum_correctness() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!((baseline_sum(&data) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn baseline_norm_correctness() {
        let data = vec![3.0f32, 4.0];
        assert!((baseline_norm(&data) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn baseline_min_float_correctness() {
        let data = vec![3.0f32, 1.0, 4.0, 1.5];
        assert!((baseline_min_float(&data) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn baseline_max_float_correctness() {
        let data = vec![3.0f32, 1.0, 4.0, 1.5];
        assert!((baseline_max_float(&data) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn baseline_argmin_float_correctness() {
        let data = vec![3.0f32, 1.0, 4.0, 1.5];
        assert_eq!(baseline_argmin_float(&data), 1);
    }

    #[test]
    fn baseline_argmax_float_correctness() {
        let data = vec![3.0f32, 1.0, 4.0, 1.5];
        assert_eq!(baseline_argmax_float(&data), 2);
    }

    #[test]
    fn baseline_moments_correctness() {
        let data = vec![2.0f32, 4.0, 6.0, 8.0];
        let (mean, variance) = baseline_moments(&data);
        assert!((mean - 5.0).abs() < 1e-6);
        assert!((variance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn baseline_minmax_float_correctness() {
        let data = vec![3.0f32, 1.0, 4.0, 1.5];
        let (min_val, min_idx, max_val, max_idx) = baseline_minmax_float(&data);
        assert!((min_val - 1.0).abs() < 1e-6);
        assert_eq!(min_idx, 1);
        assert!((max_val - 4.0).abs() < 1e-6);
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn baseline_min_ord_correctness() {
        let data = vec![3i32, 1, 4, 2];
        assert_eq!(baseline_min_ord(&data), 1);
    }

    #[test]
    fn baseline_max_ord_correctness() {
        let data = vec![3i32, 1, 4, 2];
        assert_eq!(baseline_max_ord(&data), 4);
    }

    #[test]
    fn baseline_argmin_ord_correctness() {
        let data = vec![3i32, 1, 4, 2];
        assert_eq!(baseline_argmin_ord(&data), 1);
    }

    #[test]
    fn baseline_argmax_ord_correctness() {
        let data = vec![3i32, 1, 4, 2];
        assert_eq!(baseline_argmax_ord(&data), 2);
    }

    #[test]
    fn baseline_minmax_ord_correctness() {
        let data = vec![3i32, 1, 4, 2];
        let (min_val, min_idx, max_val, max_idx) = baseline_minmax_ord(&data);
        assert_eq!(min_val, 1);
        assert_eq!(min_idx, 1);
        assert_eq!(max_val, 4);
        assert_eq!(max_idx, 2);
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

    #[test]
    fn baseline_sum_int_i8_correctness() {
        let data = vec![1i8, 2, 3, 4];
        assert_eq!(baseline_sum_int(&data), 10);
    }

    #[test]
    fn baseline_sum_int_i16_correctness() {
        let data = vec![100i16, 200, 300, 400];
        assert_eq!(baseline_sum_int(&data), 1000);
    }

    #[test]
    fn baseline_sum_int_i32_correctness() {
        let data = vec![100_000i32, 200_000, 300_000];
        assert_eq!(baseline_sum_int(&data), 600_000);
    }

    #[test]
    fn baseline_sum_int_i64_correctness() {
        let data = vec![1_000_000_000i64, 2_000_000_000, 3_000_000_000];
        assert_eq!(baseline_sum_int(&data), 6_000_000_000);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets =
        bench_sum,
        bench_norm,
        bench_min,
        bench_max,
        bench_argmin,
        bench_argmax,
        bench_moments,
        bench_minmax,
        bench_row_norms
}
criterion_main!(benches);

// endregion
