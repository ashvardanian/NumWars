//! Benchmark for ragged scaled-dot-product attention
//!
//! Computes O = softmax(Q K^T / sqrt(d)) V for a single ragged segment with 8 heads,
//! comparing NumKong's pre-packed KV-cache kernels against a composed candle CPU
//! pipeline (matmul -> softmax -> matmul, the mainstream Rust shape — candle's fused
//! flash-attention exists only for CUDA).
//!
//! Competitors:
//! - numkong (AttentionKeyValueCache::try_attention on pre-packed KV)
//! - candle (candle-core matmul + candle-nn softmax_last_dim)
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_attention --bench bench_attention
//! NUMWARS_FILTER="bf16|i8" cargo bench --features bench_attention
//! ```
//!
//! Benchmark naming: attention/{dtype}
//! Examples: attention/f32, attention/bf16, attention/i8

use std::hint::black_box;

use candle_core::{DType, Device, Tensor as CandleTensor};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use numkong::{bf16, capabilities, e4m3, Attention, AttentionKeyValueCache, Tensor};

use numwars::Settings;

const HEAD_COUNT: usize = 8;

// region: Per-library runs

fn run_numkong<T: Attention + Clone>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    init: T,
    kv_length: usize,
    head_dim: usize,
    query_count: usize,
) {
    let row_width = HEAD_COUNT * head_dim;
    let queries = Tensor::<T>::try_full(&[query_count, row_width], init.clone()).expect("Failed to create queries");
    let keys = Tensor::<T>::try_full(&[kv_length, row_width], init.clone()).expect("Failed to create keys");
    let values = Tensor::<T>::try_full(&[kv_length, row_width], init).expect("Failed to create values");
    let kv_offsets = [0u32, kv_length as u32];
    let query_offsets = [0u32, query_count as u32];
    let cache = AttentionKeyValueCache::<T>::try_pack(&keys.view(), &values.view(), head_dim, &kv_offsets, None)
        .expect("Failed to pack the KV cache");

    group.bench_function("numkong", |b| {
        b.iter(|| {
            let outputs = cache
                .try_attention(black_box(&queries.view()), &query_offsets, None)
                .expect("Attention failed");
            black_box(outputs)
        })
    });
}

fn run_candle(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dtype: DType,
    kv_length: usize,
    head_dim: usize,
    query_count: usize,
) {
    let device = Device::Cpu;
    let queries = CandleTensor::full(0.5f32, (1, HEAD_COUNT, query_count, head_dim), &device)
        .and_then(|t| t.to_dtype(dtype))
        .expect("Failed to create queries");
    let keys = CandleTensor::full(0.5f32, (1, HEAD_COUNT, kv_length, head_dim), &device)
        .and_then(|t| t.to_dtype(dtype))
        .expect("Failed to create keys");
    let values = CandleTensor::full(0.5f32, (1, HEAD_COUNT, kv_length, head_dim), &device)
        .and_then(|t| t.to_dtype(dtype))
        .expect("Failed to create values");
    let scale = 1.0 / (head_dim as f64).sqrt();

    group.bench_function("candle", |b| {
        b.iter(|| {
            let scores = black_box(&queries)
                .matmul(&keys.transpose(2, 3).expect("transpose failed"))
                .and_then(|t| t.affine(scale, 0.0))
                .expect("Q K^T failed");
            let weights = candle_nn::ops::softmax_last_dim(&scores).expect("softmax failed");
            let outputs = weights.matmul(&values).expect("P V failed");
            black_box(outputs)
        })
    });
}

// endregion

// region: Benchmarks

fn bench_attention_dtype(
    c: &mut Criterion,
    settings: &Settings,
    dtype: &str,
    kv_length: usize,
    head_dim: usize,
    query_count: usize,
    run: impl FnOnce(&mut BenchmarkGroup<'_, WallTime>),
) {
    let name = format!("attention/{dtype}");
    if !settings.selects(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    // Q·K^T and P·V each cost 2·h·q·kv·d scalar operations
    group.throughput(Throughput::Elements(
        (4 * HEAD_COUNT * query_count * kv_length * head_dim) as u64,
    ));
    run(&mut group);
    group.finish();
}

/// Benchmark ragged scaled-dot-product attention.
pub fn bench_attention(c: &mut Criterion, settings: &Settings) {
    // candle's CPU matmul fans out over the global rayon pool, which would compare a many-core
    // run against single-threaded peers. Must be set before the pool is first used.
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        std::env::set_var("RAYON_NUM_THREADS", settings.threads.to_string());
    }
    let kv_length = settings.dims_height;
    let head_dim = settings.dims_width;
    let query_count = settings.dims_depth;

    // candle's CPU backend has no BF16 matmul ("unsupported dtype BF16 for op matmul"),
    // so it only competes in the F32 group below.
    bench_attention_dtype(c, settings, "bf16", kv_length, head_dim, query_count, |group| {
        run_numkong(group, bf16::from_f32(0.5), kv_length, head_dim, query_count);
    });
    bench_attention_dtype(c, settings, "e4m3", kv_length, head_dim, query_count, |group| {
        run_numkong(group, e4m3::from_f32(0.5), kv_length, head_dim, query_count);
    });
    bench_attention_dtype(c, settings, "i8", kv_length, head_dim, query_count, |group| {
        run_numkong(group, 16i8, kv_length, head_dim, query_count);
    });
    bench_attention_dtype(c, settings, "f32", kv_length, head_dim, query_count, |group| {
        run_candle(group, DType::F32, kv_length, head_dim, query_count);
    });
}

// endregion

// region: Main

fn main() {
    let settings = Settings::read();
    capabilities::configure_thread();
    let mut criterion = numwars::configure_criterion(&settings);
    bench_attention(&mut criterion, &settings);
    criterion.final_summary();
}

// endregion
