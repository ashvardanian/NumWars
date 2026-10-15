//! Benchmark for elementwise tensor operations
//!
//! Compares NumKong vs baseline Rust implementations and selected ecosystem crates
//! for elementwise operations across supported dtypes.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_each --bench bench_each
//! NUMWARS_FILTER="each/sum|each/scale|each/fma" cargo bench --features bench_each
//! ```
//!
//! Benchmark naming: each/{operation}/{dtype}
//! Examples: each/sum/f32, each/scale/f64, each/fma/bf16

use std::hint::black_box;
use std::ops::{Add, Mul};

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use numkong::{bf16, f16, Capabilities, EachFma, EachScale, EachSum};

use numwars::Settings;

// region: Operation Model

#[derive(Clone, Copy)]
enum EachOp {
    Sum,
    Scale,
    Fma,
}

impl EachOp {
    fn slug(self) -> &'static str {
        match self {
            EachOp::Sum => "sum",
            EachOp::Scale => "scale",
            EachOp::Fma => "fma",
        }
    }

    fn input_count(self) -> usize {
        match self {
            EachOp::Scale => 1,
            EachOp::Sum => 2,
            EachOp::Fma => 3,
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

/// outᵢ = α × aᵢ × bᵢ + β × cᵢ, the contract of NumKong's `each_fma`.
fn baseline_fma<T: Add<Output = T> + Mul<Output = T> + Copy>(
    a: &[T],
    b: &[T],
    c: &[T],
    alpha: T,
    beta: T,
    out: &mut [T],
) {
    for i in 0..a.len() {
        out[i] = alpha * a[i] * b[i] + beta * c[i];
    }
}

// endregion

// region: Per-Library Run Traits

trait RunBaseline: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunBaseline for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], c: &[f32]) {
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
            EachOp::Fma => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_fma(a, b, c, 2.5f32, 0.5f32, &mut out);
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunBaseline for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], c: &[f64]) {
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
            EachOp::Fma => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_fma(a, b, c, 2.5f64, 0.5f64, &mut out);
                    black_box(&out);
                })
            }),
        };
    }
}

impl RunBaseline for i8 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[i8], b: &[i8], c: &[i8]) {
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
            EachOp::Fma => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_fma(a, b, c, 2i8, 1i8, &mut out);
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

/// Times NumKong's `op` over `a`, `b` and `c`, with the affine scalars `alpha` and `beta`.
fn run_numkong<T: EachSum + EachScale + EachFma<Scalar = <T as EachScale>::Scalar> + Clone>(
    op: EachOp,
    group: &mut BenchmarkGroup<'_, WallTime>,
    a: &[T],
    b: &[T],
    c: &[T],
    alpha: <T as EachScale>::Scalar,
    beta: <T as EachScale>::Scalar,
) where
    <T as EachScale>::Scalar: Copy,
{
    let mut out = a.to_vec();
    let _ = match op {
        EachOp::Sum => group.bench_function("numkong", |bench| {
            bench.iter(|| {
                EachSum::each_sum(a, b, &mut out).expect("each_sum failed");
                black_box(&out);
            })
        }),
        EachOp::Scale => group.bench_function("numkong", |bench| {
            bench.iter(|| {
                EachScale::each_scale(a, alpha, beta, &mut out).expect("each_scale failed");
                black_box(&out);
            })
        }),
        EachOp::Fma => group.bench_function("numkong", |bench| {
            bench.iter(|| {
                EachFma::each_fma(a, b, c, alpha, beta, &mut out).expect("each_fma failed");
                black_box(&out);
            })
        }),
    };
}

impl RunNumKong for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], c: &[f32]) {
        run_numkong(op, group, a, b, c, 2.5f32, 0.5f32);
    }
}

impl RunNumKong for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], c: &[f64]) {
        run_numkong(op, group, a, b, c, 2.5f64, 0.5f64);
    }
}

impl RunNumKong for f16 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f16], b: &[f16], c: &[f16]) {
        run_numkong(op, group, a, b, c, 2.5f32, 0.5f32);
    }
}

impl RunNumKong for bf16 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[bf16], b: &[bf16], c: &[bf16]) {
        run_numkong(op, group, a, b, c, 2.5f32, 0.5f32);
    }
}

impl RunNumKong for i8 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[i8], b: &[i8], c: &[i8]) {
        run_numkong(op, group, a, b, c, 2.5f32, 0.5f32);
    }
}

trait RunNdarray: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunNdarray for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], c: &[f32]) {
        let a_nd = ndarray::Array1::from(a.to_vec());
        let b_nd = ndarray::Array1::from(b.to_vec());
        let c_nd = ndarray::Array1::from(c.to_vec());
        let _ = match op {
            EachOp::Sum => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd + &b_nd))),
            EachOp::Scale => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * 2.5f32))),
            EachOp::Fma => group.bench_function("ndarray", |bench| {
                bench.iter(|| black_box(&a_nd * &b_nd * 2.5f32 + &c_nd * 0.5f32))
            }),
        };
    }
}

impl RunNdarray for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], c: &[f64]) {
        let a_nd = ndarray::Array1::from(a.to_vec());
        let b_nd = ndarray::Array1::from(b.to_vec());
        let c_nd = ndarray::Array1::from(c.to_vec());
        let _ = match op {
            EachOp::Sum => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd + &b_nd))),
            EachOp::Scale => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * 2.5f64))),
            EachOp::Fma => group.bench_function("ndarray", |bench| {
                bench.iter(|| black_box(&a_nd * &b_nd * 2.5f64 + &c_nd * 0.5f64))
            }),
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
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], c: &[f32]) {
        let a_na = nalgebra::DVector::from_column_slice(a);
        let b_na = nalgebra::DVector::from_column_slice(b);
        let c_na = nalgebra::DVector::from_column_slice(c);
        let _ = match op {
            EachOp::Sum => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na + &b_na))),
            EachOp::Scale => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na * 2.5f32))),
            EachOp::Fma => group.bench_function("nalgebra", |bench| {
                bench.iter(|| black_box(a_na.component_mul(&b_na) * 2.5f32 + &c_na * 0.5f32))
            }),
        };
    }
}

impl RunNalgebra for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], c: &[f64]) {
        let a_na = nalgebra::DVector::from_column_slice(a);
        let b_na = nalgebra::DVector::from_column_slice(b);
        let c_na = nalgebra::DVector::from_column_slice(c);
        let _ = match op {
            EachOp::Sum => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na + &b_na))),
            EachOp::Scale => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na * 2.5f64))),
            EachOp::Fma => group.bench_function("nalgebra", |bench| {
                bench.iter(|| black_box(a_na.component_mul(&b_na) * 2.5f64 + &c_na * 0.5f64))
            }),
        };
    }
}

impl RunNalgebra for f16 {}
impl RunNalgebra for bf16 {}
impl RunNalgebra for i8 {}

// endregion

// region: Generic Helpers

fn bench_each_op_dtype<T>(c: &mut Criterion, settings: &Settings, op: EachOp, dtype: &str, init: T)
where
    T: Clone + RunBaseline + RunNumKong + RunNdarray + RunNalgebra + 'static,
{
    let name = format!("each/{}/{}", op.slug(), dtype);
    if !settings.selects(&name) {
        return;
    }

    let size = settings.batch_size();
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

pub fn bench_sum(c: &mut Criterion, settings: &Settings) {
    bench_each_op_dtype(c, settings, EachOp::Sum, "f32", 1.0f32);
    bench_each_op_dtype(c, settings, EachOp::Sum, "f64", 1.0f64);
    bench_each_op_dtype(c, settings, EachOp::Sum, "f16", f16::from_f32(1.0));
    bench_each_op_dtype(c, settings, EachOp::Sum, "bf16", bf16::from_f32(1.0));
    bench_each_op_dtype(c, settings, EachOp::Sum, "i8", 1i8);
}

pub fn bench_scale(c: &mut Criterion, settings: &Settings) {
    bench_each_op_dtype(c, settings, EachOp::Scale, "f32", 1.0f32);
    bench_each_op_dtype(c, settings, EachOp::Scale, "f64", 1.0f64);
    bench_each_op_dtype(c, settings, EachOp::Scale, "f16", f16::from_f32(1.0));
    bench_each_op_dtype(c, settings, EachOp::Scale, "bf16", bf16::from_f32(1.0));
    bench_each_op_dtype(c, settings, EachOp::Scale, "i8", 1i8);
}

pub fn bench_fma(c: &mut Criterion, settings: &Settings) {
    bench_each_op_dtype(c, settings, EachOp::Fma, "f32", 1.0f32);
    bench_each_op_dtype(c, settings, EachOp::Fma, "f64", 1.0f64);
    bench_each_op_dtype(c, settings, EachOp::Fma, "f16", f16::from_f32(1.0));
    bench_each_op_dtype(c, settings, EachOp::Fma, "bf16", bf16::from_f32(1.0));
    bench_each_op_dtype(c, settings, EachOp::Fma, "i8", 1i8);
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

    #[test]
    fn fma_baseline_correctness() {
        let (a, b, c) = ([1.0f32, 2.0], [3.0f32, 4.0], [5.0f32, 6.0]);
        let mut out = [0.0f32; 2];
        baseline_fma(&a, &b, &c, 2.0, 0.5, &mut out);
        assert_eq!(out, [8.5, 19.0]);
    }
}

// endregion

// region: Main

fn main() {
    let settings = Settings::read("each");
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    let mut criterion = numwars::configure_criterion(&settings);
    bench_sum(&mut criterion, &settings);
    bench_scale(&mut criterion, &settings);
    bench_fma(&mut criterion, &settings);
    criterion.final_summary();
}

// endregion
