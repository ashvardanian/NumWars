//! Benchmark for MaxSim (ColBERT-style late-interaction) operations
//!
//! MaxSim computes: score = Sigma_i max_j dot(q_i, d_j)
//! Used in ColBERT and other late-interaction retrieval models.
//!
//! Competitors:
//! - numkong (MaxSimPackedMatrix::score on pre-packed tensors)
//! - ndarray (matmul-based: Q @ D.T then row-wise max + sum)
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_maxsim --bench bench_maxsim
//! NUMWARS_FILTER="f16|bf16|f32" cargo bench --features bench_maxsim
//! ```
//!
//! Benchmark naming: maxsim/{dtype}
//! Examples: maxsim/f32, maxsim/bf16, maxsim/f16

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use ndarray::Array2;
use numkong::prelude::*;
use numkong::{bf16, capabilities, f16, MaxSim, MaxSimPackedMatrix};

use numwars::Settings;

// region: Per-Library Run Traits

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
        bench.iter(|| black_box(packed_queries.try_score(&packed_documents).expect("scoring failed")))
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

// endregion

// region: Generic Helpers

fn bench_maxsim_dtype<T>(c: &mut Criterion, settings: &Settings, dtype: &str, init: T)
where
    T: Clone + RunNumKong + RunNdarray + 'static,
{
    let name = format!("maxsim/{dtype}");
    if !settings.selects(&name) {
        return;
    }

    let (query_count, document_count, dimension) = (settings.dims_height, settings.dims_width, settings.dims_depth);

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
    <T as RunNdarray>::run(
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
pub fn bench_maxsim(c: &mut Criterion, settings: &Settings) {
    bench_maxsim_dtype(c, settings, "f32", 1.0f32);
    bench_maxsim_dtype(c, settings, "bf16", bf16::from_f32(1.0));
    bench_maxsim_dtype(c, settings, "f16", f16::from_f32(1.0));
}

// endregion

// region: Main

fn main() {
    let settings = Settings::read();
    capabilities::configure_thread();
    let mut criterion = numwars::configure_criterion(&settings);
    bench_maxsim(&mut criterion, &settings);
    criterion.final_summary();
}

// endregion
