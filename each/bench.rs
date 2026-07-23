//! Benchmark for elementwise tensor operations
//!
//! Compares NumKong vs baseline Rust implementations and selected ecosystem crates
//! for elementwise operations across supported dtypes.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_each --bench bench_each
//! NUMWARS_FILTER="each/sum|each/scale" cargo bench --features bench_each
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Tensor size in elements (default: 1000000)
//! - NUMWARS_FILTER: Regex to filter benchmark names
//!
//! Benchmark naming: each/{operation}/{dtype}
//! Examples: each/sum/f32, each/scale/f64

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use numkong::{bf16, capabilities, f16, EachScale, EachSum};
use std::hint::black_box;
use std::ops::{Add, Mul};
use utils::*;

// region: Operation Model

#[derive(Clone, Copy)]
enum EachOp {
    Sum,
    Scale,
}

impl EachOp {
    fn slug(self) -> &'static str {
        match self {
            EachOp::Sum => "sum",
            EachOp::Scale => "scale",
        }
    }

    fn input_count(self) -> usize {
        match self {
            EachOp::Scale => 1,
            EachOp::Sum => 2,
        }
    }
}

// endregion

// region: Baseline Kernels

fn baseline_sum<T: Add<Output = T> + Copy>(a: &[T], b: &[T], out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

fn baseline_scale<T: Mul<Output = T> + Copy>(a: &[T], alpha: T, out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i] * alpha;
    }
}

// endregion

// region: Per-Library Run Traits

trait RunBaseline: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunBaseline for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], _c: &[f32]) {
        let mut out = vec![0.0f32; a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_sum(a, b, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_scale(a, 2.5f32, &mut out);
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunBaseline for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], _c: &[f64]) {
        let mut out = vec![0.0f64; a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_sum(a, b, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_scale(a, 2.5f64, &mut out);
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunBaseline for i8 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[i8], b: &[i8], _c: &[i8]) {
        let mut out = vec![0i8; a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_sum(a, b, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_scale(a, 2i8, &mut out);
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunBaseline for f16 {}
impl RunBaseline for bf16 {}

trait RunNumKong: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunNumKong for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], _c: &[f32]) {
        let mut out = vec![0.0f32; a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSum::each_sum(a, b, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachScale::each_scale(a, 2.5f32, 0.0f32, &mut out));
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunNumKong for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], _c: &[f64]) {
        let mut out = vec![0.0f64; a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSum::each_sum(a, b, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachScale::each_scale(a, 2.5f64, 0.0f64, &mut out));
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunNumKong for f16 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f16], b: &[f16], _c: &[f16]) {
        let mut out = vec![f16::from_f32(0.0); a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSum::each_sum(a, b, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachScale::each_scale(a, 2.5f32, 0.0f32, &mut out));
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunNumKong for bf16 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[bf16], b: &[bf16], _c: &[bf16]) {
        let mut out = vec![bf16::from_f32(0.0); a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSum::each_sum(a, b, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachScale::each_scale(a, 2.5f32, 0.0f32, &mut out));
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunNumKong for i8 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[i8], b: &[i8], _c: &[i8]) {
        let mut out = vec![0i8; a.len()];
        let _ = match op {
            EachOp::Sum => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSum::each_sum(a, b, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachScale::each_scale(a, 2.5f32, 0.0f32, &mut out));
                    black_box(&out);
                })
            }),
        };
    }
}

trait RunNdarray: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunNdarray for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], _c: &[f32]) {
        let a_nd = ndarray::Array1::from(a.to_vec());
        let b_nd = ndarray::Array1::from(b.to_vec());
        let _ = match op {
            EachOp::Sum => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd + &b_nd))),
            EachOp::Scale => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * 2.5f32))),
        };
    }
}

impl RunNdarray for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], _c: &[f64]) {
        let a_nd = ndarray::Array1::from(a.to_vec());
        let b_nd = ndarray::Array1::from(b.to_vec());
        let _ = match op {
            EachOp::Sum => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd + &b_nd))),
            EachOp::Scale => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * 2.5f64))),
        };
    }
}

impl RunNdarray for f16 {}
impl RunNdarray for bf16 {}
impl RunNdarray for i8 {}

trait RunNalgebra: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunNalgebra for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], _c: &[f32]) {
        let a_na = nalgebra::DVector::from_column_slice(a);
        let b_na = nalgebra::DVector::from_column_slice(b);
        let _ = match op {
            EachOp::Sum => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na + &b_na))),
            EachOp::Scale => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na * 2.5f32))),
        };
    }
}

impl RunNalgebra for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], _c: &[f64]) {
        let a_na = nalgebra::DVector::from_column_slice(a);
        let b_na = nalgebra::DVector::from_column_slice(b);
        let _ = match op {
            EachOp::Sum => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na + &b_na))),
            EachOp::Scale => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na * 2.5f64))),
        };
    }
}

impl RunNalgebra for f16 {}
impl RunNalgebra for bf16 {}
impl RunNalgebra for i8 {}

// endregion

// region: Generic Helpers

fn bench_each_op_dtype<T>(c: &mut Criterion, op: EachOp, dtype: &str, size: usize, init: T)
where
    T: Clone + RunBaseline + RunNumKong + RunNdarray + RunNalgebra + 'static,
{
    let name = format!("each/{}/{}", op.slug(), dtype);
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(
        (op.input_count() * size * std::mem::size_of::<T>()) as u64,
    ));

    let a = vec![init.clone(); size];
    let b = vec![init.clone(); size];
    let c_data = vec![init; size];

    <T as RunNumKong>::run(op, &mut group, &a, &b, &c_data);
    <T as RunBaseline>::run(op, &mut group, &a, &b, &c_data);
    <T as RunNdarray>::run(op, &mut group, &a, &b, &c_data);
    <T as RunNalgebra>::run(op, &mut group, &a, &b, &c_data);

    group.finish();
}

// endregion

// region: Entry Points

pub fn bench_sum(c: &mut Criterion) {
    capabilities::configure_thread();
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Sum, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Sum, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Sum, "f16", size, f16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Sum, "bf16", size, bf16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Sum, "i8", size, 1i8);
}

pub fn bench_scale(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Scale, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Scale, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Scale, "f16", size, f16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Scale, "bf16", size, bf16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Scale, "i8", size, 1i8);
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn sum_baseline_correctness() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut out = vec![0.0f32; 3];
        baseline_sum(&a, &b, &mut out);
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn scale_baseline_correctness() {
        let a = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        baseline_scale(&a, 2.0f32, &mut out);
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_sum, bench_scale
}
criterion_main!(benches);

// endregion
