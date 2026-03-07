//! Benchmark for elementwise tensor operations
//!
//! Compares NumKong vs baseline Rust implementations and selected ecosystem crates
//! for elementwise operations across supported dtypes.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_each --bench bench_each
//! NUMWARS_FILTER="each/sum|each/mul" cargo bench --features bench_each
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Tensor size in elements (default: 1000000)
//! - NUMWARS_FILTER: Regex to filter benchmark names
//!
//! Benchmark naming: each/{operation}/{dtype}
//! Examples: each/sum/f32, each/mul/f64, each/blend/i8

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use num_traits::Float;
use numkong::{bf16, f16, EachATan, EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum};
use std::hint::black_box;
use std::ops::{Add, Mul};
use utils::*;

// region: Operation model

#[derive(Clone, Copy)]
enum EachOp {
    Sum,
    Mul,
    Scale,
    Blend,
    Fma,
    Sin,
    Cos,
    Atan,
}

impl EachOp {
    fn slug(self) -> &'static str {
        match self {
            EachOp::Sum => "sum",
            EachOp::Mul => "mul",
            EachOp::Scale => "scale",
            EachOp::Blend => "blend",
            EachOp::Fma => "fma",
            EachOp::Sin => "sin",
            EachOp::Cos => "cos",
            EachOp::Atan => "atan",
        }
    }

    fn input_count(self) -> usize {
        match self {
            EachOp::Fma => 3,
            EachOp::Scale | EachOp::Sin | EachOp::Cos | EachOp::Atan => 1,
            EachOp::Sum | EachOp::Mul | EachOp::Blend => 2,
        }
    }
}

// endregion

// region: Baseline kernels

fn baseline_sum<T: Add<Output = T> + Copy>(a: &[T], b: &[T], out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

fn baseline_mul<T: Mul<Output = T> + Copy>(a: &[T], b: &[T], out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

fn baseline_scale<T: Mul<Output = T> + Copy>(a: &[T], alpha: T, out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i] * alpha;
    }
}

fn baseline_blend<T: Add<Output = T> + Mul<Output = T> + Copy>(a: &[T], b: &[T], alpha: T, beta: T, out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = alpha * a[i] + beta * b[i];
    }
}

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

fn baseline_sin<T: Float>(a: &[T], out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i].sin();
    }
}

fn baseline_cos<T: Float>(a: &[T], out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i].cos();
    }
}

fn baseline_atan<T: Float>(a: &[T], out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i].atan();
    }
}

// endregion

// region: Backend run traits

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
            EachOp::Mul => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_mul(a, b, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_scale(a, 2.5f32, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Blend => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_blend(a, b, 1.5f32, 2.5f32, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_fma(a, b, c, 1.5f32, 2.5f32, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Sin => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_sin(a, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Cos => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_cos(a, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Atan => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_atan(a, &mut out);
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
            EachOp::Mul => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_mul(a, b, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_scale(a, 2.5f64, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Blend => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_blend(a, b, 1.5f64, 2.5f64, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_fma(a, b, c, 1.5f64, 2.5f64, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Sin => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_sin(a, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Cos => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_cos(a, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Atan => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_atan(a, &mut out);
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
            EachOp::Mul => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_mul(a, b, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Scale => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_scale(a, 2i8, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Blend => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_blend(a, b, 1i8, 2i8, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("baseline", |bench| {
                bench.iter(|| {
                    baseline_fma(a, b, c, 1i8, 2i8, &mut out);
                    black_box(&out);
                })
            }),
            EachOp::Sin | EachOp::Cos | EachOp::Atan => group,
        };
    }
}

impl RunBaseline for f16 {}
impl RunBaseline for bf16 {}

trait RunNumKong: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunNumKong for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], b: &[f32], c: &[f32]) {
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
            EachOp::Blend => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachBlend::each_blend(a, b, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachFMA::each_fma(a, b, c, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Sin => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSin::sin(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Cos => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachCos::cos(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Atan => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachATan::atan(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Mul => group,
        };
    }
}

impl RunNumKong for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], c: &[f64]) {
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
            EachOp::Blend => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachBlend::each_blend(a, b, 1.5f64, 2.5f64, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachFMA::each_fma(a, b, c, 1.5f64, 2.5f64, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Sin => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSin::sin(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Cos => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachCos::cos(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Atan => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachATan::atan(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Mul => group,
        };
    }
}

impl RunNumKong for f16 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f16], b: &[f16], c: &[f16]) {
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
            EachOp::Blend => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachBlend::each_blend(a, b, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachFMA::each_fma(a, b, c, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Sin => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachSin::sin(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Cos => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachCos::cos(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Atan => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachATan::atan(a, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Mul => group,
        };
    }
}

impl RunNumKong for bf16 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[bf16], b: &[bf16], c: &[bf16]) {
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
            EachOp::Blend => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachBlend::each_blend(a, b, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachFMA::each_fma(a, b, c, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Mul | EachOp::Sin | EachOp::Cos | EachOp::Atan => group,
        };
    }
}

impl RunNumKong for i8 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[i8], b: &[i8], c: &[i8]) {
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
            EachOp::Blend => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachBlend::each_blend(a, b, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Fma => group.bench_function("numkong", |bench| {
                bench.iter(|| {
                    black_box(EachFMA::each_fma(a, b, c, 1.5f32, 2.5f32, &mut out));
                    black_box(&out);
                })
            }),
            EachOp::Mul | EachOp::Sin | EachOp::Cos | EachOp::Atan => group,
        };
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
            EachOp::Mul => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * &b_nd))),
            EachOp::Scale => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * 2.5f32))),
            EachOp::Blend => group.bench_function("ndarray", |bench| {
                bench.iter(|| black_box(&a_nd * 1.5f32 + &b_nd * 2.5f32))
            }),
            EachOp::Fma => group.bench_function("ndarray", |bench| {
                bench.iter(|| black_box(&a_nd * &b_nd * 1.5f32 + &c_nd * 2.5f32))
            }),
            EachOp::Sin => group.bench_function("ndarray", |bench| bench.iter(|| black_box(a_nd.mapv(f32::sin)))),
            EachOp::Cos => group.bench_function("ndarray", |bench| bench.iter(|| black_box(a_nd.mapv(f32::cos)))),
            EachOp::Atan => group.bench_function("ndarray", |bench| bench.iter(|| black_box(a_nd.mapv(f32::atan)))),
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
            EachOp::Mul => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * &b_nd))),
            EachOp::Scale => group.bench_function("ndarray", |bench| bench.iter(|| black_box(&a_nd * 2.5f64))),
            EachOp::Blend => group.bench_function("ndarray", |bench| {
                bench.iter(|| black_box(&a_nd * 1.5f64 + &b_nd * 2.5f64))
            }),
            EachOp::Fma => group.bench_function("ndarray", |bench| {
                bench.iter(|| black_box(&a_nd * &b_nd * 1.5f64 + &c_nd * 2.5f64))
            }),
            EachOp::Sin => group.bench_function("ndarray", |bench| bench.iter(|| black_box(a_nd.mapv(f64::sin)))),
            EachOp::Cos => group.bench_function("ndarray", |bench| bench.iter(|| black_box(a_nd.mapv(f64::cos)))),
            EachOp::Atan => group.bench_function("ndarray", |bench| bench.iter(|| black_box(a_nd.mapv(f64::atan)))),
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
            EachOp::Mul => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(a_na.component_mul(&b_na))))
            }
            EachOp::Scale => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na * 2.5f32))),
            EachOp::Blend | EachOp::Fma | EachOp::Sin | EachOp::Cos | EachOp::Atan => group,
        };
    }
}

impl RunNalgebra for f64 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f64], b: &[f64], _c: &[f64]) {
        let a_na = nalgebra::DVector::from_column_slice(a);
        let b_na = nalgebra::DVector::from_column_slice(b);
        let _ = match op {
            EachOp::Sum => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na + &b_na))),
            EachOp::Mul => {
                group.bench_function("nalgebra", |bench| bench.iter(|| black_box(a_na.component_mul(&b_na))))
            }
            EachOp::Scale => group.bench_function("nalgebra", |bench| bench.iter(|| black_box(&a_na * 2.5f64))),
            EachOp::Blend | EachOp::Fma | EachOp::Sin | EachOp::Cos | EachOp::Atan => group,
        };
    }
}

impl RunNalgebra for f16 {}
impl RunNalgebra for bf16 {}
impl RunNalgebra for i8 {}

trait RunWide: Sized {
    fn run(_op: EachOp, _group: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _c: &[Self]) {}
}

impl RunWide for f32 {
    fn run(op: EachOp, group: &mut BenchmarkGroup<'_, WallTime>, a: &[f32], _b: &[f32], _c: &[f32]) {
        match op {
            EachOp::Sin => {
                let chunks: Vec<wide::f32x8> = a
                    .chunks_exact(8)
                    .map(|chunk| wide::f32x8::from(<[f32; 8]>::try_from(chunk).expect("chunk size mismatch")))
                    .collect();
                let mut out = vec![wide::f32x8::ZERO; chunks.len()];
                group.bench_function("wide", |bench| {
                    bench.iter(|| {
                        for (i, chunk) in chunks.iter().enumerate() {
                            out[i] = chunk.sin();
                        }
                        black_box(&out);
                    })
                });
            }
            EachOp::Cos => {
                let chunks: Vec<wide::f32x8> = a
                    .chunks_exact(8)
                    .map(|chunk| wide::f32x8::from(<[f32; 8]>::try_from(chunk).expect("chunk size mismatch")))
                    .collect();
                let mut out = vec![wide::f32x8::ZERO; chunks.len()];
                group.bench_function("wide", |bench| {
                    bench.iter(|| {
                        for (i, chunk) in chunks.iter().enumerate() {
                            out[i] = chunk.cos();
                        }
                        black_box(&out);
                    })
                });
            }
            EachOp::Sum | EachOp::Mul | EachOp::Scale | EachOp::Blend | EachOp::Fma | EachOp::Atan => {}
        }
    }
}

impl RunWide for f64 {}
impl RunWide for f16 {}
impl RunWide for bf16 {}
impl RunWide for i8 {}

// endregion

// region: Generic helper

fn bench_each_op_dtype<T>(c: &mut Criterion, op: EachOp, dtype: &str, size: usize, init: T)
where
    T: Clone + RunBaseline + RunNumKong + RunNdarray + RunNalgebra + RunWide + 'static,
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
    <T as RunWide>::run(op, &mut group, &a, &b, &c_data);

    group.finish();
}

// endregion

// region: Entry points

pub fn bench_sum(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Sum, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Sum, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Sum, "f16", size, f16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Sum, "bf16", size, bf16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Sum, "i8", size, 1i8);
}

pub fn bench_mul(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Mul, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Mul, "f64", size, 1.0f64);
}

pub fn bench_scale(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Scale, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Scale, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Scale, "f16", size, f16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Scale, "bf16", size, bf16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Scale, "i8", size, 1i8);
}

pub fn bench_blend(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Blend, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Blend, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Blend, "f16", size, f16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Blend, "bf16", size, bf16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Blend, "i8", size, 1i8);
}

pub fn bench_fma(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Fma, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Fma, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Fma, "f16", size, f16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Fma, "bf16", size, bf16::from_f32(1.0));
    bench_each_op_dtype(c, EachOp::Fma, "i8", size, 1i8);
}

pub fn bench_sin(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Sin, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Sin, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Sin, "f16", size, f16::from_f32(1.0));
}

pub fn bench_cos(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Cos, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Cos, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Cos, "f16", size, f16::from_f32(1.0));
}

pub fn bench_atan(c: &mut Criterion) {
    let size = get_tensor_dims();
    bench_each_op_dtype(c, EachOp::Atan, "f32", size, 1.0f32);
    bench_each_op_dtype(c, EachOp::Atan, "f64", size, 1.0f64);
    bench_each_op_dtype(c, EachOp::Atan, "f16", size, f16::from_f32(1.0));
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
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
    fn mul_baseline_correctness() {
        let a = vec![2.0f32, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0];
        let mut out = vec![0.0f32; 3];
        baseline_mul(&a, &b, &mut out);
        assert_eq!(out, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn scale_baseline_correctness() {
        let a = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        baseline_scale(&a, 2.0f32, &mut out);
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn blend_baseline_correctness() {
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];
        let mut out = vec![0.0f32; 2];
        baseline_blend(&a, &b, 2.0, 3.0, &mut out);
        assert_eq!(out, vec![11.0, 16.0]);
    }

    #[test]
    fn fma_baseline_correctness() {
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];
        let c = vec![5.0f32, 6.0];
        let mut out = vec![0.0f32; 2];
        baseline_fma(&a, &b, &c, 2.0, 3.0, &mut out);
        assert_eq!(out, vec![21.0, 34.0]);
    }

    #[test]
    fn sin_baseline_correctness() {
        let a = vec![0.0f32, std::f32::consts::FRAC_PI_2];
        let mut out = vec![0.0f32; 2];
        baseline_sin(&a, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cos_baseline_correctness() {
        let a = vec![0.0f32, std::f32::consts::PI];
        let mut out = vec![0.0f32; 2];
        baseline_cos(&a, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn atan_baseline_correctness() {
        let a = vec![0.0f32, 1.0];
        let mut out = vec![0.0f32; 2];
        baseline_atan(&a, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_sum, bench_mul, bench_scale, bench_blend, bench_fma, bench_sin, bench_cos, bench_atan
}
criterion_main!(benches);

// endregion
