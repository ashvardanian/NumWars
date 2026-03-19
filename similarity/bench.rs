//! Benchmark for vector similarity operations
//!
//! Compares NumKong vs baseline Rust implementations for pairwise similarity metrics.
//! Benchmarks spatial (dot, angular, euclidean, sqeuclidean), binary (hamming, jaccard), and probability
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
//! - NUMWARS_DIMS: Vector dimensions (default: 2048)
//! - NUMWARS_FILTER: Regex to filter benchmark names (default: none, runs all)
//! - NUMWARS_WARMUP_SECONDS: Warmup time in seconds (default: 3.0)
//! - NUMWARS_PROFILE_SECONDS: Measurement time in seconds (default: 10.0)
//!
//! Benchmark naming: similarity/{metric}/{dtype}
//! Examples: similarity/angular/f32, similarity/dot/f64, similarity/euclidean/i8

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use num_traits::{Float, Num, NumCast};
use numkong::{
    bf16, capabilities, e2m3, e3m2, e4m3, e5m2, f16, u1x8, Angular, Dot, Euclidean, Hamming, Jaccard, JensenShannon,
    KullbackLeibler,
};
use std::hint::black_box;
use std::iter::Sum;
use std::ops::AddAssign;
use std::os::raw::c_int;
use utils::*;

// region: Baseline Implementations

const UNROLL: usize = 8;

#[link(name = "blas")]
unsafe extern "C" {
    fn cblas_sdot(n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int) -> f32;
    fn cblas_ddot(n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int) -> f64;
}

trait BaselineDotType: BaselineConvert<Self::Acc> + Copy + Sized + 'static {
    type Acc: Num + Copy + NumCast + AddAssign + 'static;
    type Output: NumCast + Copy + 'static;
}

trait BaselineSqEuclideanType: BaselineConvert<Self::Acc> + Copy + Sized + 'static {
    type Acc: Num + Copy + NumCast + AddAssign + 'static;
    type Output: NumCast + Copy + 'static;
}

trait BaselineEuclideanType: BaselineConvert<Self::Acc> + Copy + Sized + 'static {
    type Acc: Num + Copy + NumCast + AddAssign + 'static;
    type Output: Float + NumCast + Copy + 'static;
}

trait BaselineAngularType: BaselineConvert<Self::Acc> + Copy + Sized + 'static {
    type Acc: Num + Copy + NumCast + AddAssign + 'static;
    type Output: Float + NumCast + Copy + 'static;
}

impl BaselineDotType for f32 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineDotType for f64 {
    type Acc = f64;
    type Output = f64;
}
impl BaselineDotType for i8 {
    type Acc = i32;
    type Output = i32;
}
impl BaselineDotType for u8 {
    type Acc = u32;
    type Output = u32;
}
impl BaselineDotType for f16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineDotType for bf16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineDotType for e4m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineDotType for e5m2 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineDotType for e2m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineDotType for e3m2 {
    type Acc = f32;
    type Output = f32;
}

impl BaselineSqEuclideanType for f32 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineSqEuclideanType for f64 {
    type Acc = f64;
    type Output = f64;
}
impl BaselineSqEuclideanType for i8 {
    type Acc = i32;
    type Output = u32;
}
impl BaselineSqEuclideanType for u8 {
    type Acc = i32;
    type Output = u32;
}
impl BaselineSqEuclideanType for f16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineSqEuclideanType for bf16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineSqEuclideanType for e4m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineSqEuclideanType for e5m2 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineSqEuclideanType for e2m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineSqEuclideanType for e3m2 {
    type Acc = f32;
    type Output = f32;
}

impl BaselineEuclideanType for f32 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineEuclideanType for f64 {
    type Acc = f64;
    type Output = f64;
}
impl BaselineEuclideanType for i8 {
    type Acc = i32;
    type Output = f32;
}
impl BaselineEuclideanType for u8 {
    type Acc = i32;
    type Output = f32;
}
impl BaselineEuclideanType for f16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineEuclideanType for bf16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineEuclideanType for e4m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineEuclideanType for e5m2 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineEuclideanType for e2m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineEuclideanType for e3m2 {
    type Acc = f32;
    type Output = f32;
}

impl BaselineAngularType for f32 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineAngularType for f64 {
    type Acc = f64;
    type Output = f64;
}
impl BaselineAngularType for i8 {
    type Acc = i32;
    type Output = f32;
}
impl BaselineAngularType for u8 {
    type Acc = u32;
    type Output = f32;
}
impl BaselineAngularType for f16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineAngularType for bf16 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineAngularType for e4m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineAngularType for e5m2 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineAngularType for e2m3 {
    type Acc = f32;
    type Output = f32;
}
impl BaselineAngularType for e3m2 {
    type Acc = f32;
    type Output = f32;
}

trait UnrolledReducer<Acc> {
    type Output;
    fn step8(&mut self, a: [Acc; UNROLL], b: [Acc; UNROLL]);
    fn step1(&mut self, a: Acc, b: Acc);
    fn finish(self) -> Self::Output;
}

#[inline(always)]
fn sum8<Acc: Num + Copy>(lanes: [Acc; UNROLL]) -> Acc {
    lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7]
}

fn reduce_unrolled8<T, Acc, R>(a: &[T], b: &[T], mut reducer: R) -> Option<R::Output>
where
    T: BaselineConvert<Acc>,
    Acc: Num + Copy + NumCast + AddAssign + 'static,
    R: UnrolledReducer<Acc>,
{
    if a.len() != b.len() {
        return None;
    }

    let mut a_chunks = a.chunks_exact(UNROLL);
    let mut b_chunks = b.chunks_exact(UNROLL);

    for (a8, b8) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let a_values = [
            a8[0].to_acc(),
            a8[1].to_acc(),
            a8[2].to_acc(),
            a8[3].to_acc(),
            a8[4].to_acc(),
            a8[5].to_acc(),
            a8[6].to_acc(),
            a8[7].to_acc(),
        ];
        let b_values = [
            b8[0].to_acc(),
            b8[1].to_acc(),
            b8[2].to_acc(),
            b8[3].to_acc(),
            b8[4].to_acc(),
            b8[5].to_acc(),
            b8[6].to_acc(),
            b8[7].to_acc(),
        ];
        reducer.step8(a_values, b_values);
    }

    for (&a_value, &b_value) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        reducer.step1(a_value.to_acc(), b_value.to_acc());
    }

    Some(reducer.finish())
}

struct DotReducer<Acc> {
    lanes: [Acc; UNROLL],
}

impl<Acc> DotReducer<Acc>
where
    Acc: Num + Copy + NumCast + AddAssign + 'static,
{
    fn new() -> Self {
        Self {
            lanes: [Acc::zero(); UNROLL],
        }
    }
}

impl<Acc> UnrolledReducer<Acc> for DotReducer<Acc>
where
    Acc: Num + Copy + NumCast + AddAssign + 'static,
{
    type Output = Acc;

    fn step8(&mut self, a: [Acc; UNROLL], b: [Acc; UNROLL]) {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = b;
        self.lanes[0] += a0 * b0;
        self.lanes[1] += a1 * b1;
        self.lanes[2] += a2 * b2;
        self.lanes[3] += a3 * b3;
        self.lanes[4] += a4 * b4;
        self.lanes[5] += a5 * b5;
        self.lanes[6] += a6 * b6;
        self.lanes[7] += a7 * b7;
    }

    fn step1(&mut self, a: Acc, b: Acc) {
        self.lanes[0] += a * b;
    }

    fn finish(self) -> Self::Output {
        sum8(self.lanes)
    }
}

struct SqEuclideanReducer<Acc> {
    lanes: [Acc; UNROLL],
}

impl<Acc> SqEuclideanReducer<Acc>
where
    Acc: Num + Copy + NumCast + AddAssign + 'static,
{
    fn new() -> Self {
        Self {
            lanes: [Acc::zero(); UNROLL],
        }
    }
}

impl<Acc> UnrolledReducer<Acc> for SqEuclideanReducer<Acc>
where
    Acc: Num + Copy + NumCast + AddAssign + 'static,
{
    type Output = Acc;

    fn step8(&mut self, a: [Acc; UNROLL], b: [Acc; UNROLL]) {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = b;
        let d0 = a0 - b0;
        let d1 = a1 - b1;
        let d2 = a2 - b2;
        let d3 = a3 - b3;
        let d4 = a4 - b4;
        let d5 = a5 - b5;
        let d6 = a6 - b6;
        let d7 = a7 - b7;
        self.lanes[0] += d0 * d0;
        self.lanes[1] += d1 * d1;
        self.lanes[2] += d2 * d2;
        self.lanes[3] += d3 * d3;
        self.lanes[4] += d4 * d4;
        self.lanes[5] += d5 * d5;
        self.lanes[6] += d6 * d6;
        self.lanes[7] += d7 * d7;
    }

    fn step1(&mut self, a: Acc, b: Acc) {
        let diff = a - b;
        self.lanes[0] += diff * diff;
    }

    fn finish(self) -> Self::Output {
        sum8(self.lanes)
    }
}

struct AngularStats<Acc> {
    dot: Acc,
    norm_a: Acc,
    norm_b: Acc,
}

struct AngularReducer<Acc> {
    dot_lanes: [Acc; UNROLL],
    norm_a: [Acc; 2],
    norm_b: [Acc; 2],
}

impl<Acc> AngularReducer<Acc>
where
    Acc: Num + Copy + NumCast + AddAssign + 'static,
{
    fn new() -> Self {
        Self {
            dot_lanes: [Acc::zero(); UNROLL],
            norm_a: [Acc::zero(); 2],
            norm_b: [Acc::zero(); 2],
        }
    }
}

impl<Acc> UnrolledReducer<Acc> for AngularReducer<Acc>
where
    Acc: Num + Copy + NumCast + AddAssign + 'static,
{
    type Output = AngularStats<Acc>;

    fn step8(&mut self, a: [Acc; UNROLL], b: [Acc; UNROLL]) {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = b;

        self.dot_lanes[0] += a0 * b0;
        self.dot_lanes[1] += a1 * b1;
        self.dot_lanes[2] += a2 * b2;
        self.dot_lanes[3] += a3 * b3;
        self.dot_lanes[4] += a4 * b4;
        self.dot_lanes[5] += a5 * b5;
        self.dot_lanes[6] += a6 * b6;
        self.dot_lanes[7] += a7 * b7;

        self.norm_a[0] += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        self.norm_a[1] += a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
        self.norm_b[0] += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
        self.norm_b[1] += b4 * b4 + b5 * b5 + b6 * b6 + b7 * b7;
    }

    fn step1(&mut self, a: Acc, b: Acc) {
        self.dot_lanes[0] += a * b;
        self.norm_a[0] += a * a;
        self.norm_b[0] += b * b;
    }

    fn finish(self) -> Self::Output {
        AngularStats {
            dot: sum8(self.dot_lanes),
            norm_a: self.norm_a[0] + self.norm_a[1],
            norm_b: self.norm_b[0] + self.norm_b[1],
        }
    }
}

fn baseline_dot<T: BaselineDotType>(a: &[T], b: &[T]) -> Option<T::Output> {
    reduce_unrolled8(a, b, DotReducer::<T::Acc>::new()).map(|value| NumCast::from(value).unwrap())
}

fn baseline_sqeuclidean<T: BaselineSqEuclideanType>(a: &[T], b: &[T]) -> Option<T::Output> {
    reduce_unrolled8(a, b, SqEuclideanReducer::<T::Acc>::new()).map(|value| NumCast::from(value).unwrap())
}

fn baseline_euclidean<T: BaselineEuclideanType>(a: &[T], b: &[T]) -> Option<T::Output> {
    reduce_unrolled8(a, b, SqEuclideanReducer::<T::Acc>::new()).map(|value| {
        let squared_distance: T::Output = NumCast::from(value).unwrap();
        squared_distance.sqrt()
    })
}

fn baseline_angular<T: BaselineAngularType>(a: &[T], b: &[T]) -> Option<T::Output> {
    reduce_unrolled8(a, b, AngularReducer::<T::Acc>::new()).map(|stats| {
        let dot: T::Output = NumCast::from(stats.dot).unwrap();
        let norm_a: T::Output = NumCast::from(stats.norm_a).unwrap();
        let norm_b: T::Output = NumCast::from(stats.norm_b).unwrap();
        let one: T::Output = NumCast::from(1.0).unwrap();
        one - (dot / (norm_a.sqrt() * norm_b.sqrt()))
    })
}

fn baseline_hamming_u1x8(a: &[u1x8], b: &[u1x8]) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .map(|(byte_a, byte_b)| (byte_a.bits() ^ byte_b.bits()).count_ones())
            .sum(),
    )
}

fn baseline_jaccard_u1x8(a: &[u1x8], b: &[u1x8]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let (intersection_count, union_count) =
        a.iter()
            .zip(b)
            .fold((0u32, 0u32), |(intersection, union), (byte_a, byte_b)| {
                let bits_a = byte_a.bits();
                let bits_b = byte_b.bits();
                (
                    intersection + (bits_a & bits_b).count_ones(),
                    union + (bits_a | bits_b).count_ones(),
                )
            });

    if union_count == 0 {
        return Some(0.0);
    }

    Some(1.0 - (intersection_count as f32 / union_count as f32))
}

fn baseline_kullbackleibler<T: Float + Sum>(a: &[T], b: &[T]) -> Option<T> {
    if a.len() != b.len() {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .map(|(&value_a, &value_b)| value_a * (value_a / value_b).ln())
            .sum(),
    )
}

fn baseline_jensenshannon<T: Float + Sum>(a: &[T], b: &[T]) -> Option<T> {
    if a.len() != b.len() {
        return None;
    }
    let one_half = T::from(0.5).unwrap();
    let (divergence_a_mid, divergence_b_mid) = a.iter().zip(b).fold(
        (T::zero(), T::zero()),
        |(divergence_a, divergence_b), (&value_a, &value_b)| {
            let midpoint = (value_a + value_b) * one_half;
            (
                divergence_a + value_a * (value_a / midpoint).ln(),
                divergence_b + value_b * (value_b / midpoint).ln(),
            )
        },
    );
    Some(((divergence_a_mid + divergence_b_mid) * one_half).sqrt())
}

// endregion

// region: Per-library Run traits

trait RunBaselineAngular: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: BaselineAngularType> RunBaselineAngular for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_angular(a, b))));
    }
}

trait RunNumKongAngular: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: Angular> RunNumKongAngular for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("numkong", |bench| bench.iter(|| black_box(Angular::angular(a, b))));
    }
}

trait RunBaselineSqEuclidean: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: BaselineSqEuclideanType> RunBaselineSqEuclidean for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_sqeuclidean(a, b))));
    }
}

trait RunNumKongSqEuclidean: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: Euclidean> RunNumKongSqEuclidean for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(Euclidean::sqeuclidean(a, b)))
        });
    }
}

trait RunBaselineEuclidean: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: BaselineEuclideanType> RunBaselineEuclidean for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_euclidean(a, b))));
    }
}

trait RunNumKongEuclidean: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: Euclidean> RunNumKongEuclidean for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("numkong", |bench| bench.iter(|| black_box(Euclidean::euclidean(a, b))));
    }
}

trait RunBaselineDot: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: BaselineDotType> RunBaselineDot for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("baseline", |bench| bench.iter(|| black_box(baseline_dot(a, b))));
    }
}

trait RunNumKongDot: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: Dot> RunNumKongDot for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("numkong", |bench| bench.iter(|| black_box(Dot::dot(a, b))));
    }
}

trait RunNalgebraDot: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _dims: usize, _v: Self) {}
}
impl RunNalgebraDot for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f32) {
        let a = nalgebra::DVector::<f32>::from_element(dims, v);
        let b = nalgebra::DVector::<f32>::from_element(dims, v);
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(a.dot(&b))));
    }
}
impl RunNalgebraDot for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f64) {
        let a = nalgebra::DVector::<f64>::from_element(dims, v);
        let b = nalgebra::DVector::<f64>::from_element(dims, v);
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box(a.dot(&b))));
    }
}
impl RunNalgebraDot for i8 {}
impl RunNalgebraDot for u8 {}
impl RunNalgebraDot for f16 {}
impl RunNalgebraDot for bf16 {}
impl RunNalgebraDot for e4m3 {}
impl RunNalgebraDot for e5m2 {}
impl RunNalgebraDot for e2m3 {}
impl RunNalgebraDot for e3m2 {}

trait RunNdarrayDot: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _dims: usize, _v: Self) {}
}
impl RunNdarrayDot for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f32) {
        let a = ndarray::Array1::<f32>::from_elem(dims, v);
        let b = ndarray::Array1::<f32>::from_elem(dims, v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&b))));
    }
}
impl RunNdarrayDot for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f64) {
        let a = ndarray::Array1::<f64>::from_elem(dims, v);
        let b = ndarray::Array1::<f64>::from_elem(dims, v);
        group.bench_function("ndarray", |bench| bench.iter(|| black_box(a.dot(&b))));
    }
}
impl RunNdarrayDot for i8 {}
impl RunNdarrayDot for u8 {}
impl RunNdarrayDot for f16 {}
impl RunNdarrayDot for bf16 {}
impl RunNdarrayDot for e4m3 {}
impl RunNdarrayDot for e5m2 {}
impl RunNdarrayDot for e2m3 {}
impl RunNdarrayDot for e3m2 {}

trait RunBlasDot: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _dims: usize, _v: Self) {}
}
impl RunBlasDot for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f32) {
        let n = i32::try_from(dims).expect("dims exceed cblas integer range");
        let a = vec![v; dims];
        let b = vec![v; dims];
        group.bench_function("blas", |bench| {
            bench.iter(|| unsafe { black_box(cblas_sdot(n, a.as_ptr(), 1, b.as_ptr(), 1)) })
        });
    }
}
impl RunBlasDot for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f64) {
        let n = i32::try_from(dims).expect("dims exceed cblas integer range");
        let a = vec![v; dims];
        let b = vec![v; dims];
        group.bench_function("blas", |bench| {
            bench.iter(|| unsafe { black_box(cblas_ddot(n, a.as_ptr(), 1, b.as_ptr(), 1)) })
        });
    }
}
impl RunBlasDot for i8 {}
impl RunBlasDot for u8 {}
impl RunBlasDot for f16 {}
impl RunBlasDot for bf16 {}
impl RunBlasDot for e4m3 {}
impl RunBlasDot for e5m2 {}
impl RunBlasDot for e2m3 {}
impl RunBlasDot for e3m2 {}

trait RunNalgebraEuclidean: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _dims: usize, _v: Self) {}
}
impl RunNalgebraEuclidean for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f32) {
        let a = nalgebra::DVector::<f32>::from_element(dims, v);
        let b = nalgebra::DVector::<f32>::from_element(dims, v);
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box((&a - &b).norm())));
    }
}
impl RunNalgebraEuclidean for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f64) {
        let a = nalgebra::DVector::<f64>::from_element(dims, v);
        let b = nalgebra::DVector::<f64>::from_element(dims, v);
        group.bench_function("nalgebra", |bench| bench.iter(|| black_box((&a - &b).norm())));
    }
}
impl RunNalgebraEuclidean for i8 {}
impl RunNalgebraEuclidean for u8 {}
impl RunNalgebraEuclidean for f16 {}
impl RunNalgebraEuclidean for bf16 {}
impl RunNalgebraEuclidean for e4m3 {}
impl RunNalgebraEuclidean for e5m2 {}
impl RunNalgebraEuclidean for e2m3 {}
impl RunNalgebraEuclidean for e3m2 {}

trait RunNdarrayEuclidean: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _dims: usize, _v: Self) {}
}
impl RunNdarrayEuclidean for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f32) {
        let a = ndarray::Array1::<f32>::from_elem(dims, v);
        let b = ndarray::Array1::<f32>::from_elem(dims, v);
        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let diff = &a - &b;
                black_box(diff.dot(&diff).sqrt())
            })
        });
    }
}
impl RunNdarrayEuclidean for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, dims: usize, v: f64) {
        let a = ndarray::Array1::<f64>::from_elem(dims, v);
        let b = ndarray::Array1::<f64>::from_elem(dims, v);
        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let diff = &a - &b;
                black_box(diff.dot(&diff).sqrt())
            })
        });
    }
}
impl RunNdarrayEuclidean for i8 {}
impl RunNdarrayEuclidean for u8 {}
impl RunNdarrayEuclidean for f16 {}
impl RunNdarrayEuclidean for bf16 {}
impl RunNdarrayEuclidean for e4m3 {}
impl RunNdarrayEuclidean for e5m2 {}
impl RunNdarrayEuclidean for e2m3 {}
impl RunNdarrayEuclidean for e3m2 {}

trait RunBaselineKullbackLeibler: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: Float + Sum> RunBaselineKullbackLeibler for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("baseline", |bench| {
            bench.iter(|| black_box(baseline_kullbackleibler(a, b)))
        });
    }
}

trait RunNumKongKullbackLeibler: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: KullbackLeibler> RunNumKongKullbackLeibler for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(KullbackLeibler::kullbackleibler(a, b)))
        });
    }
}

trait RunBaselineJensenShannon: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: Float + Sum> RunBaselineJensenShannon for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("baseline", |bench| {
            bench.iter(|| black_box(baseline_jensenshannon(a, b)))
        });
    }
}

trait RunNumKongJensenShannon: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self]) {}
}
impl<T: JensenShannon> RunNumKongJensenShannon for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a: &[T], b: &[T]) {
        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(JensenShannon::jensenshannon(a, b)))
        });
    }
}

// endregion

// region: Generic helpers

fn bench_angular_dtype<T>(c: &mut Criterion, dtype: &str, dims: usize, init: T)
where
    T: Clone + RunBaselineAngular + RunNumKongAngular + 'static,
{
    let name = format!("similarity/angular/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((2 * dims * std::mem::size_of::<T>()) as u64));
    let a = vec![init.clone(); dims];
    let b = vec![init; dims];
    <T as RunNumKongAngular>::run(&mut group, &a, &b);
    <T as RunBaselineAngular>::run(&mut group, &a, &b);
    group.finish();
}

fn bench_sqeuclidean_dtype<T>(c: &mut Criterion, dtype: &str, dims: usize, init: T)
where
    T: Clone + RunBaselineSqEuclidean + RunNumKongSqEuclidean + 'static,
{
    let name = format!("similarity/sqeuclidean/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((2 * dims * std::mem::size_of::<T>()) as u64));
    let a = vec![init.clone(); dims];
    let b = vec![init; dims];
    <T as RunNumKongSqEuclidean>::run(&mut group, &a, &b);
    <T as RunBaselineSqEuclidean>::run(&mut group, &a, &b);
    group.finish();
}

fn bench_euclidean_dtype<T>(c: &mut Criterion, dtype: &str, dims: usize, init: T)
where
    T: Clone + RunBaselineEuclidean + RunNumKongEuclidean + RunNalgebraEuclidean + RunNdarrayEuclidean + 'static,
{
    let name = format!("similarity/euclidean/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((2 * dims * std::mem::size_of::<T>()) as u64));
    let extra_init = init.clone();
    let a = vec![init.clone(); dims];
    let b = vec![init; dims];
    <T as RunNumKongEuclidean>::run(&mut group, &a, &b);
    <T as RunBaselineEuclidean>::run(&mut group, &a, &b);
    <T as RunNalgebraEuclidean>::run(&mut group, dims, extra_init.clone());
    <T as RunNdarrayEuclidean>::run(&mut group, dims, extra_init);
    group.finish();
}

fn bench_dot_dtype<T>(c: &mut Criterion, dtype: &str, dims: usize, init: T)
where
    T: Clone + RunBaselineDot + RunNumKongDot + RunNalgebraDot + RunNdarrayDot + RunBlasDot + 'static,
{
    let name = format!("similarity/dot/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((2 * dims * std::mem::size_of::<T>()) as u64));
    let extra_init = init.clone();
    let a = vec![init.clone(); dims];
    let b = vec![init; dims];
    <T as RunNumKongDot>::run(&mut group, &a, &b);
    <T as RunBaselineDot>::run(&mut group, &a, &b);
    <T as RunNalgebraDot>::run(&mut group, dims, extra_init.clone());
    <T as RunNdarrayDot>::run(&mut group, dims, extra_init.clone());
    <T as RunBlasDot>::run(&mut group, dims, extra_init);
    group.finish();
}

fn bench_kullbackleibler_dtype<T>(c: &mut Criterion, dtype: &str, dims: usize, init: T)
where
    T: Clone + RunBaselineKullbackLeibler + RunNumKongKullbackLeibler + 'static,
{
    let name = format!("similarity/kullbackleibler/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((dims * std::mem::size_of::<T>()) as u64));
    let a = vec![init.clone(); dims];
    let b = vec![init; dims];
    <T as RunNumKongKullbackLeibler>::run(&mut group, &a, &b);
    <T as RunBaselineKullbackLeibler>::run(&mut group, &a, &b);
    group.finish();
}

fn bench_jensenshannon_dtype<T>(c: &mut Criterion, dtype: &str, dims: usize, init: T)
where
    T: Clone + RunBaselineJensenShannon + RunNumKongJensenShannon + 'static,
{
    let name = format!("similarity/jensenshannon/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((dims * std::mem::size_of::<T>()) as u64));
    let a = vec![init.clone(); dims];
    let b = vec![init; dims];
    <T as RunNumKongJensenShannon>::run(&mut group, &a, &b);
    <T as RunBaselineJensenShannon>::run(&mut group, &a, &b);
    group.finish();
}

// endregion

// region: Benchmarks

/// Benchmark true Euclidean distance
pub fn bench_euclidean(c: &mut Criterion) {
    capabilities::configure_thread();
    let dims = get_vector_dims();
    bench_euclidean_dtype(c, "f32", dims, 1.0f32);
    bench_euclidean_dtype(c, "f64", dims, 1.0f64);
    bench_euclidean_dtype(c, "i8", dims, 1i8);
    bench_euclidean_dtype(c, "u8", dims, 1u8);
    bench_euclidean_dtype(c, "f16", dims, f16::from_f32(1.0));
    bench_euclidean_dtype(c, "bf16", dims, bf16::from_f32(1.0));
    bench_euclidean_dtype(c, "e4m3", dims, e4m3::from_f32(1.0));
    bench_euclidean_dtype(c, "e5m2", dims, e5m2::from_f32(1.0));
    bench_euclidean_dtype(c, "e2m3", dims, e2m3::from_f32(1.0));
    bench_euclidean_dtype(c, "e3m2", dims, e3m2::from_f32(1.0));
}

/// Benchmark squared Euclidean distance
pub fn bench_sqeuclidean(c: &mut Criterion) {
    let dims = get_vector_dims();
    bench_sqeuclidean_dtype(c, "f32", dims, 1.0f32);
    bench_sqeuclidean_dtype(c, "f64", dims, 1.0f64);
    bench_sqeuclidean_dtype(c, "i8", dims, 1i8);
    bench_sqeuclidean_dtype(c, "u8", dims, 1u8);
    bench_sqeuclidean_dtype(c, "f16", dims, f16::from_f32(1.0));
    bench_sqeuclidean_dtype(c, "bf16", dims, bf16::from_f32(1.0));
    bench_sqeuclidean_dtype(c, "e4m3", dims, e4m3::from_f32(1.0));
    bench_sqeuclidean_dtype(c, "e5m2", dims, e5m2::from_f32(1.0));
    bench_sqeuclidean_dtype(c, "e2m3", dims, e2m3::from_f32(1.0));
    bench_sqeuclidean_dtype(c, "e3m2", dims, e3m2::from_f32(1.0));
}

/// Benchmark angular distance
pub fn bench_angular(c: &mut Criterion) {
    let dims = get_vector_dims();
    bench_angular_dtype(c, "f32", dims, 1.0f32);
    bench_angular_dtype(c, "f64", dims, 1.0f64);
    bench_angular_dtype(c, "i8", dims, 1i8);
    bench_angular_dtype(c, "u8", dims, 1u8);
    bench_angular_dtype(c, "f16", dims, f16::from_f32(1.0));
    bench_angular_dtype(c, "bf16", dims, bf16::from_f32(1.0));
    bench_angular_dtype(c, "e4m3", dims, e4m3::from_f32(1.0));
    bench_angular_dtype(c, "e5m2", dims, e5m2::from_f32(1.0));
    bench_angular_dtype(c, "e2m3", dims, e2m3::from_f32(1.0));
    bench_angular_dtype(c, "e3m2", dims, e3m2::from_f32(1.0));
}

/// Benchmark dot product
pub fn bench_dot(c: &mut Criterion) {
    let dims = get_vector_dims();
    bench_dot_dtype(c, "f32", dims, 1.0f32);
    bench_dot_dtype(c, "f64", dims, 1.0f64);
    bench_dot_dtype(c, "i8", dims, 1i8);
    bench_dot_dtype(c, "u8", dims, 1u8);
    bench_dot_dtype(c, "f16", dims, f16::from_f32(1.0));
    bench_dot_dtype(c, "bf16", dims, bf16::from_f32(1.0));
    bench_dot_dtype(c, "e4m3", dims, e4m3::from_f32(1.0));
    bench_dot_dtype(c, "e5m2", dims, e5m2::from_f32(1.0));
    bench_dot_dtype(c, "e2m3", dims, e2m3::from_f32(1.0));
    bench_dot_dtype(c, "e3m2", dims, e3m2::from_f32(1.0));
}

/// Benchmark Hamming distance
pub fn bench_hamming(c: &mut Criterion) {
    let dims = get_vector_dims();
    let byte_count = dims.div_ceil(8);
    if should_run_benchmark("similarity/hamming/u1x8") {
        let mut group = c.benchmark_group("similarity/hamming/u1x8");
        let vector_a = vec![u1x8::new(0xAA); byte_count];
        let vector_b = vec![u1x8::new(0x55); byte_count];
        group.throughput(Throughput::Bytes(byte_count as u64));
        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(Hamming::hamming(&vector_a, &vector_b)))
        });
        group.bench_function("baseline", |bench| {
            bench.iter(|| black_box(baseline_hamming_u1x8(&vector_a, &vector_b)))
        });
        group.finish();
    }
}

/// Benchmark Jaccard distance
pub fn bench_jaccard(c: &mut Criterion) {
    let dims = get_vector_dims();
    let byte_count = dims.div_ceil(8);
    if should_run_benchmark("similarity/jaccard/u1x8") {
        let mut group = c.benchmark_group("similarity/jaccard/u1x8");
        let vector_a = vec![u1x8::new(0xAA); byte_count];
        let vector_b = vec![u1x8::new(0x55); byte_count];
        group.throughput(Throughput::Bytes(byte_count as u64));
        group.bench_function("numkong", |bench| {
            bench.iter(|| black_box(Jaccard::jaccard(&vector_a, &vector_b)))
        });
        group.bench_function("baseline", |bench| {
            bench.iter(|| black_box(baseline_jaccard_u1x8(&vector_a, &vector_b)))
        });
        group.finish();
    }
}

/// Benchmark Kullback-Leibler divergence
pub fn bench_kullbackleibler(c: &mut Criterion) {
    let dims = get_vector_dims();
    bench_kullbackleibler_dtype(c, "f32", dims, 1.0f32);
    bench_kullbackleibler_dtype(c, "f64", dims, 1.0f64);
}

/// Benchmark Jensen-Shannon divergence
pub fn bench_jensenshannon(c: &mut Criterion) {
    let dims = get_vector_dims();
    bench_jensenshannon_dtype(c, "f32", dims, 1.0f32);
    bench_jensenshannon_dtype(c, "f64", dims, 1.0f64);
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_baseline_correctness_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        let result = baseline_dot(&a, &b);
        assert!(result.is_some());
        assert!((result.unwrap() - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn dot_baseline_correctness_i8_output_type() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];

        let result: Option<i32> = baseline_dot(&a, &b);
        assert_eq!(result, Some(32));
    }

    #[test]
    fn dot_baseline_correctness_u8_output_type() {
        let a = vec![1u8, 2, 3];
        let b = vec![4u8, 5, 6];

        let result: Option<u32> = baseline_dot(&a, &b);
        assert_eq!(result, Some(32));
    }

    #[test]
    fn sqeuclidean_baseline_correctness_f32() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];

        let result_sq = baseline_sqeuclidean(&a, &b);
        assert!(result_sq.is_some());
        let result = result_sq.unwrap().sqrt();
        assert!((result - 5.0).abs() < 1e-6); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn sqeuclidean_baseline_correctness_i8_output_type() {
        let a = vec![0i8, 0];
        let b = vec![3i8, 4];

        let result: Option<u32> = baseline_sqeuclidean(&a, &b);
        assert_eq!(result, Some(25));
    }

    #[test]
    fn euclidean_baseline_correctness_u8_output_type() {
        let a = vec![0u8, 0];
        let b = vec![3u8, 4];

        let result: Option<f32> = baseline_euclidean(&a, &b);
        assert_eq!(result, Some(5.0));
    }

    #[test]
    fn hamming_baseline_correctness() {
        let a = vec![u1x8::new(0b11110000)];
        let b = vec![u1x8::new(0b10100000)];
        // XOR = 0b01010000, popcount = 2
        assert_eq!(baseline_hamming_u1x8(&a, &b), Some(2));
    }

    #[test]
    fn jaccard_baseline_correctness() {
        let a = vec![u1x8::new(0b11110000)];
        let b = vec![u1x8::new(0b11000000)];
        // intersection = 0b11000000, popcount = 2
        // union = 0b11110000, popcount = 4
        // jaccard = 1 - 2/4 = 0.5
        assert!((baseline_jaccard_u1x8(&a, &b).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn kullbackleibler_baseline_correctness() {
        let a = vec![0.5f32, 0.5];
        let b = vec![0.5f32, 0.5];
        // KL(P||P) = 0
        assert!(baseline_kullbackleibler(&a, &b).unwrap().abs() < 1e-6);
    }

    #[test]
    fn jensenshannon_baseline_correctness() {
        let a = vec![0.5f32, 0.5];
        let b = vec![0.5f32, 0.5];
        // JS(P, P) = 0
        assert!(baseline_jensenshannon(&a, &b).unwrap().abs() < 1e-6);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_dot, bench_angular, bench_euclidean, bench_sqeuclidean,
              bench_hamming, bench_jaccard,
              bench_kullbackleibler, bench_jensenshannon
}
criterion_main!(benches);

// endregion
