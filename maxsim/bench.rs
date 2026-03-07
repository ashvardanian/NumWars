//! Benchmark for MaxSim (ColBERT-style late-interaction) operations
//!
//! MaxSim computes: score = Sigma_i max_j dot(q_i, d_j)
//! Used in ColBERT and other late-interaction retrieval models.
//!
//! Competitors:
//! - numkong (MaxSimPackedMatrix::score on pre-packed tensors)
//! - baseline (nested loops with scalar dot products)
//! - ndarray (matmul-based: Q @ D.T then row-wise max + sum)
//! - nalgebra (matmul-based: Q * D.T then row-wise max + sum)
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_maxsim --bench bench_maxsim
//! NUMWARS_FILTER="f16|bf16|f32" cargo bench --features bench_maxsim
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Vector dimension (default: 1536)
//! - NUMWARS_BATCH_SIZE: Number of query/document vectors (default: 1000)
//! - NUMWARS_FILTER: Regex to filter benchmark names
//!
//! Benchmark naming: maxsim/{dtype}
//! Examples: maxsim/f32, maxsim/bf16, maxsim/f16

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use nalgebra::DMatrix;
use ndarray::Array2;
use numkong::{bf16, capabilities, f16, MaxSim, MaxSimPackedMatrix, Tensor};
use std::hint::black_box;
use utils::*;

// region: Baseline implementations

fn baseline_maxsim_f32<T: BaselineConvert<f32> + Copy>(
    query_vectors: &[T],
    document_vectors: &[T],
    query_count: usize,
    document_count: usize,
    dimension: usize,
) -> f32 {
    let mut total_score = 0.0f32;
    for query_index in 0..query_count {
        let query = &query_vectors[query_index * dimension..(query_index + 1) * dimension];
        let mut max_similarity = f32::NEG_INFINITY;
        for document_index in 0..document_count {
            let document = &document_vectors[document_index * dimension..(document_index + 1) * dimension];
            let mut dot_product = 0.0f32;
            for i in 0..dimension {
                dot_product += query[i].to_acc() * document[i].to_acc();
            }
            if dot_product > max_similarity {
                max_similarity = dot_product;
            }
        }
        total_score += max_similarity;
    }
    total_score
}

// endregion

// region: Per-library run traits

trait RunNumKong: Sized {
    fn run(
        _group: &mut BenchmarkGroup<'_, WallTime>,
        _query_vectors: &[Self],
        _document_vectors: &[Self],
        _qc: usize,
        _dc: usize,
        _dim: usize,
    ) {
    }
}

fn run_numkong<T: MaxSim + Clone>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    query_vectors: &[T],
    document_vectors: &[T],
    qc: usize,
    dc: usize,
    dim: usize,
) {
    let query_tensor = Tensor::<T>::try_from_slice(query_vectors, &[qc, dim]).expect("Failed to create query tensor");
    let document_tensor =
        Tensor::<T>::try_from_slice(document_vectors, &[dc, dim]).expect("Failed to create document tensor");
    let packed_queries = MaxSimPackedMatrix::<T>::try_pack(&query_tensor.view()).expect("Failed to pack queries");
    let packed_documents =
        MaxSimPackedMatrix::<T>::try_pack(&document_tensor.view()).expect("Failed to pack documents");
    group.bench_function("numkong", |bench| {
        bench.iter(|| black_box(packed_queries.score(&packed_documents)))
    });
}

impl<T: MaxSim + Clone> RunNumKong for T {
    fn run(
        group: &mut BenchmarkGroup<'_, WallTime>,
        query_vectors: &[Self],
        document_vectors: &[Self],
        qc: usize,
        dc: usize,
        dim: usize,
    ) {
        run_numkong(group, query_vectors, document_vectors, qc, dc, dim);
    }
}

trait RunBaseline: Sized {
    fn run(
        _group: &mut BenchmarkGroup<'_, WallTime>,
        _query_vectors: &[Self],
        _document_vectors: &[Self],
        _qc: usize,
        _dc: usize,
        _dim: usize,
    ) {
    }
}

impl<T: BaselineConvert<f32> + Copy> RunBaseline for T {
    fn run(
        group: &mut BenchmarkGroup<'_, WallTime>,
        query_vectors: &[Self],
        document_vectors: &[Self],
        qc: usize,
        dc: usize,
        dim: usize,
    ) {
        group.bench_function("baseline", |bench| {
            bench.iter(|| black_box(baseline_maxsim_f32(query_vectors, document_vectors, qc, dc, dim)))
        });
    }
}

trait RunNdarray: Sized {
    fn run(
        _group: &mut BenchmarkGroup<'_, WallTime>,
        _query_vectors: &[Self],
        _document_vectors: &[Self],
        _qc: usize,
        _dc: usize,
        _dim: usize,
    ) {
    }
}

impl RunNdarray for f32 {
    fn run(
        group: &mut BenchmarkGroup<'_, WallTime>,
        query_vectors: &[f32],
        document_vectors: &[f32],
        qc: usize,
        dc: usize,
        dim: usize,
    ) {
        let query_matrix =
            Array2::from_shape_vec((qc, dim), query_vectors.to_vec()).expect("Failed to build query matrix");
        let document_matrix =
            Array2::from_shape_vec((dc, dim), document_vectors.to_vec()).expect("Failed to build document matrix");
        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores_matrix = query_matrix.dot(&document_matrix.t());
                let total: f32 = scores_matrix
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                    .sum();
                black_box(total)
            })
        });
    }
}

impl RunNdarray for f16 {}
impl RunNdarray for bf16 {}

trait RunNalgebra: Sized {
    fn run(
        _group: &mut BenchmarkGroup<'_, WallTime>,
        _query_vectors: &[Self],
        _document_vectors: &[Self],
        _qc: usize,
        _dc: usize,
        _dim: usize,
    ) {
    }
}

impl RunNalgebra for f32 {
    fn run(
        group: &mut BenchmarkGroup<'_, WallTime>,
        query_vectors: &[f32],
        document_vectors: &[f32],
        qc: usize,
        dc: usize,
        dim: usize,
    ) {
        let query_matrix = DMatrix::from_row_slice(qc, dim, query_vectors);
        let document_matrix = DMatrix::from_row_slice(dc, dim, document_vectors);
        let document_t = document_matrix.transpose();
        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores_matrix = &query_matrix * &document_t;
                let total: f32 = (0..qc)
                    .map(|i| {
                        scores_matrix
                            .row(i)
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a })
                    })
                    .sum();
                black_box(total)
            })
        });
    }
}

impl RunNalgebra for f16 {}
impl RunNalgebra for bf16 {}

// endregion

// region: Generic helper

fn bench_maxsim_dtype<T>(
    c: &mut Criterion,
    dtype: &str,
    query_count: usize,
    document_count: usize,
    dimension: usize,
    init: T,
) where
    T: Clone + RunNumKong + RunBaseline + RunNdarray + RunNalgebra + 'static,
{
    let name = format!("maxsim/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    let query_vectors = vec![init.clone(); query_count * dimension];
    let document_vectors = vec![init; document_count * dimension];
    group.throughput(Throughput::Bytes(
        ((query_count + document_count) * dimension * std::mem::size_of::<T>()) as u64,
    ));

    <T as RunNumKong>::run(
        &mut group,
        &query_vectors,
        &document_vectors,
        query_count,
        document_count,
        dimension,
    );
    <T as RunBaseline>::run(
        &mut group,
        &query_vectors,
        &document_vectors,
        query_count,
        document_count,
        dimension,
    );
    <T as RunNdarray>::run(
        &mut group,
        &query_vectors,
        &document_vectors,
        query_count,
        document_count,
        dimension,
    );
    <T as RunNalgebra>::run(
        &mut group,
        &query_vectors,
        &document_vectors,
        query_count,
        document_count,
        dimension,
    );

    group.finish();
}

// endregion

// region: Benchmarks

/// Benchmark MaxSim scoring.
pub fn bench_maxsim(c: &mut Criterion) {
    capabilities::configure_thread();
    let dimension = get_vector_dims();
    let query_count = 32; // typical ColBERT query length
    let document_count = 128; // typical ColBERT document length

    bench_maxsim_dtype(c, "f32", query_count, document_count, dimension, 1.0f32);
    bench_maxsim_dtype(c, "bf16", query_count, document_count, dimension, bf16::from_f32(1.0));
    bench_maxsim_dtype(c, "f16", query_count, document_count, dimension, f16::from_f32(1.0));
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxsim_baseline_f32_known_value() {
        // 2 queries × 2 docs × dim=2
        let queries = vec![1.0f32, 0.0, 0.0, 1.0];
        let documents = vec![1.0f32, 1.0, 0.0, 1.0];
        // q0=[1,0]: dot(q0,d0)=1, dot(q0,d1)=0 → max=1
        // q1=[0,1]: dot(q1,d0)=1, dot(q1,d1)=1 → max=1
        // total = 2
        let result = baseline_maxsim_f32(&queries, &documents, 2, 2, 2);
        assert!((result - 2.0).abs() < 1e-6);
    }

    #[test]
    fn maxsim_baseline_bf16_known_value() {
        let queries = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
        ];
        let documents = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
        ];
        let result = baseline_maxsim_f32(&queries, &documents, 2, 2, 2);
        assert!((result - 2.0).abs() < 1e-3);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_maxsim
}
criterion_main!(benches);

// endregion
