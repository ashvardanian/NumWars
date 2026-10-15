//! Benchmark for ragged scaled-dot-product attention
//!
//! Computes O = softmax(Q K^T / sqrt(d)) V for a single ragged segment with 8 heads,
//! comparing NumKong's pre-packed KV-cache kernels against a composed candle CPU
//! pipeline (matmul -> softmax -> matmul, the mainstream Rust shape — candle's fused
//! flash-attention exists only for CUDA).
//!
//! Competitors:
//! - numkong (AttentionPackedMatrix::attention on pre-packed KV)
//! - candle (candle-core matmul + candle-nn softmax_last_dim)
//!
//! A `cuda` build adds a group for each GPU in NUMWARS_DEVICES and each dtype NumKong's CUDA
//! attention takes, named like `attention/bf16<cuda:0>`. Its operands are random and live in
//! unified memory, every round synchronizes the stream, and the first round is compared with
//! NumKong's CPU attention on the leading queries. The BF16 and F16 groups race a cuBLASLt
//! composition shaped like candle's: `Q × Kᵀ` per head, a softmax kernel, then `P × V` per head.
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
use numkong::{bf16, e4m3, f16, Attention, AttentionPackedMatrix, Capabilities, Tensor};

use numwars::{try_spawn_pool, Settings};

#[cfg(feature = "cuda")]
use std::ffi::{c_void, CString};

#[cfg(feature = "cuda")]
use cudarc::{cublaslt::sys::cudaDataType, driver::result as driver};
#[cfg(feature = "cuda")]
use numkong::{NumberLike, UnifiedAllocator};
#[cfg(feature = "cuda")]
use numwars::{gpu_check, gpu_devices, CublasLtGemm, CublasLtLayout, CublasLtTypes, GpuDevice, GpuScalar};

const HEAD_COUNT: usize = 8;

// region: Per-library runs

fn run_numkong<T: Attention + Clone>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    init: T,
    kv_length: usize,
    head_dim: usize,
    query_count: usize,
    threads: usize,
) {
    let row_width = HEAD_COUNT * head_dim;
    let queries = Tensor::<T>::full(&[query_count, row_width], init.clone()).expect("Failed to create queries");
    let keys = Tensor::<T>::full(&[kv_length, row_width], init.clone()).expect("Failed to create keys");
    let values = Tensor::<T>::full(&[kv_length, row_width], init).expect("Failed to create values");
    let kv_offsets = [0u32, kv_length as u32];
    let query_offsets = [0u32, query_count as u32];

    // Bidirectional, like candle's unmasked softmax
    match try_spawn_pool(threads) {
        Some(mut pool) => {
            let cache = AttentionPackedMatrix::<T>::new_parallel(
                &keys.view(),
                &values.view(),
                head_dim,
                &kv_offsets,
                None,
                &mut pool,
            )
            .expect("Failed to pack the KV cache");
            group.bench_function("numkong", |b| {
                b.iter(|| {
                    let outputs = cache
                        .attention_parallel(
                            black_box(&queries.view()),
                            &query_offsets,
                            None,
                            usize::MAX,
                            usize::MAX,
                            &mut pool,
                        )
                        .expect("Attention failed");
                    black_box(outputs)
                })
            });
        }
        None => {
            let cache = AttentionPackedMatrix::<T>::new(&keys.view(), &values.view(), head_dim, &kv_offsets, None)
                .expect("Failed to pack the KV cache");
            group.bench_function("numkong", |b| {
                b.iter(|| {
                    let outputs = cache
                        .attention(black_box(&queries.view()), &query_offsets, None, usize::MAX, usize::MAX)
                        .expect("Attention failed");
                    black_box(outputs)
                })
            });
        }
    }
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
    let threads = settings.threads;

    // candle's CPU backend has no BF16 matmul ("unsupported dtype BF16 for op matmul"),
    // so it competes only in the F16 and F32 groups.
    bench_attention_dtype(c, settings, "bf16", kv_length, head_dim, query_count, |group| {
        run_numkong(group, bf16::from_f32(0.5), kv_length, head_dim, query_count, threads);
    });
    bench_attention_dtype(c, settings, "f16", kv_length, head_dim, query_count, |group| {
        run_numkong(group, f16::from_f32(0.5), kv_length, head_dim, query_count, threads);
        run_candle(group, DType::F16, kv_length, head_dim, query_count);
    });
    bench_attention_dtype(c, settings, "e4m3", kv_length, head_dim, query_count, |group| {
        run_numkong(group, e4m3::from_f32(0.5), kv_length, head_dim, query_count, threads);
    });
    bench_attention_dtype(c, settings, "i8", kv_length, head_dim, query_count, |group| {
        run_numkong(group, 16i8, kv_length, head_dim, query_count, threads);
    });
    bench_attention_dtype(c, settings, "f32", kv_length, head_dim, query_count, |group| {
        run_candle(group, DType::F32, kv_length, head_dim, query_count);
    });
}

// endregion

// region: GPU Rows

/// Leading queries a GPU result's first round is compared on.
#[cfg(feature = "cuda")]
const CHECKED_ROWS: usize = 16;

/// Relative error a GPU row may show against the CPU, which sums the keys in another order.
#[cfg(feature = "cuda")]
const GPU_TOLERANCE: f64 = 1e-3;

/// Relative error a GPU row may show against the CPU when the CPU rounds the probabilities to BF16,
/// as it does for `bf16` and `e4m3` inputs, so two roundings that fall differently still agree.
#[cfg(feature = "cuda")]
const BF16_PROBABILITIES_TOLERANCE: f64 = 1.0 / 64.0;

/// The rival's softmax between its two cuBLASLt products, over each row of scores in place and in
/// the input precision, as candle's CPU composition keeps them: BF16 when `BRAIN` is set, else F16.
#[cfg(feature = "cuda")]
const SOFTMAX: &str = r#"
__device__ float widen(unsigned short bits) {
#if BRAIN
    return __uint_as_float((unsigned)bits << 16);
#else
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
#endif
}
__device__ unsigned short narrow(float value) {
#if BRAIN
    unsigned bits = __float_as_uint(value);
    return (unsigned short)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
#else
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
#endif
}
__device__ float reduce(float value, bool maximum, float *shared) {
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, value, offset);
        value = maximum ? fmaxf(value, other) : value + other;
    }
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = value;
    __syncthreads();
    value = shared[0];
    for (unsigned warp = 1; warp < blockDim.x / 32; ++warp)
        value = maximum ? fmaxf(value, shared[warp]) : value + shared[warp];
    __syncthreads();
    return value;
}
extern "C" __global__ void softmax(unsigned short *scores, unsigned long long columns, float scale) {
    __shared__ float shared[32];
    unsigned short *row = scores + blockIdx.x * columns;
    float maximum = -3.4e38f, sum = 0;
    for (unsigned long long column = threadIdx.x; column < columns; column += blockDim.x)
        maximum = fmaxf(maximum, widen(row[column]));
    maximum = reduce(maximum, true, shared);
    for (unsigned long long column = threadIdx.x; column < columns; column += blockDim.x)
        sum += __expf((widen(row[column]) - maximum) * scale);
    float const inverse = 1 / reduce(sum, false, shared);
    for (unsigned long long column = threadIdx.x; column < columns; column += blockDim.x)
        row[column] = narrow(__expf((widen(row[column]) - maximum) * scale) * inverse);
}
"#;

/// A rival added to a GPU group, given the queries, keys, values, depth and the CPU's leading rows.
#[cfg(feature = "cuda")]
type Rival<T> =
    fn(&mut BenchmarkGroup<'_, WallTime>, &str, &GpuDevice, [&Tensor<T, UnifiedAllocator>; 3], usize, &[f32]);

/// NumKong's CPU attention for the leading queries against every key, the reference each GPU
/// row's first round is compared with.
#[cfg(feature = "cuda")]
fn cpu_leading_rows<T: GpuScalar + Attention>(
    [queries, keys, values]: [&Tensor<T, UnifiedAllocator>; 3],
    depth: usize,
) -> Vec<f32> {
    let (rows, width, key_count) = (
        queries.shape()[0].min(CHECKED_ROWS),
        queries.shape()[1],
        keys.shape()[0],
    );
    let leading = Tensor::<T>::from_slice(&queries.as_slice()[..rows * width], &[rows, width]).expect("Failed to copy");
    let cache = AttentionPackedMatrix::<T>::new(keys, values, depth, &[0, key_count as u32], None)
        .expect("Failed to pack on the CPU");
    let outputs = cache.attention(&leading, &[0, rows as u32], None, usize::MAX, usize::MAX);
    outputs.expect("CPU attention failed").as_slice().to_vec()
}

/// Adds the cuBLASLt composition, `Q × Kᵀ` per head, the softmax kernel, then `P × V` per head,
/// all in the input precision, to `group` as `cublaslt`, marking a first round off `expected`.
#[cfg(feature = "cuda")]
fn bench_cublaslt<T: GpuScalar + NumberLike>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    device: &GpuDevice,
    [queries, keys, values]: [&Tensor<T, UnifiedAllocator>; 3],
    depth: usize,
    expected: &[f32],
) {
    let Some(CublasLtTypes {
        input, compute, scale, ..
    }) = T::CUBLASLT
    else {
        return;
    };
    let types = CublasLtTypes {
        input,
        output: input,
        compute,
        scale,
    };
    let (query_count, width, key_count) = (queries.shape()[0], queries.shape()[1], keys.shape()[0]);
    let heads = width / depth;
    let scores_layout = CublasLtLayout {
        rows: query_count,
        columns: key_count,
        depth,
        b_transposed: true,
        row_strides: [width, width, key_count],
        batches: heads,
        batch_strides: [depth, depth, query_count * key_count],
    };
    let outputs_layout = CublasLtLayout {
        rows: query_count,
        columns: depth,
        depth: key_count,
        b_transposed: false,
        row_strides: [key_count, width, width],
        batches: heads,
        batch_strides: [query_count * key_count, depth, depth],
    };
    let gemms = CublasLtGemm::new(device, types, scores_layout)
        .and_then(|scores| Ok((scores, CublasLtGemm::new(device, types, outputs_layout)?)));
    let (scores_gemm, outputs_gemm) = match gemms {
        Ok(gemms) => gemms,
        Err(error) => return println!("{name}/cublaslt is unavailable: {error}"),
    };
    let brain = (input == cudaDataType::CUDA_R_16BF) as u8;
    let ptx = cudarc::nvrtc::compile_ptx(format!("#define BRAIN {brain}\n{SOFTMAX}")).expect("NVRTC failed");
    let image = CString::new(ptx.to_src()).expect("PTX holds no zero bytes");
    let module = unsafe { driver::module::load_data(image.as_ptr().cast()) }.expect("Failed to load the softmax");
    let entry = CString::new("softmax").expect("A plain name");
    let softmax = unsafe { driver::module::get_function(module, entry) }.expect("No softmax in the module");
    let mut scores =
        Tensor::<T, _>::zeros_in(&[heads * query_count, key_count], device.allocator()).expect("Failed to allocate");
    let mut outputs = Tensor::<T, _>::zeros_in(&[query_count, width], device.allocator()).expect("Failed to allocate");
    let launch = |scores: &mut Tensor<T, UnifiedAllocator>, outputs: &mut Tensor<T, UnifiedAllocator>| {
        scores_gemm.launch(queries, keys, scores);
        let (target, columns, scale) = (scores.as_mut_ptr(), key_count as u64, 1.0 / (depth as f32).sqrt());
        let mut arguments = [
            (&target as *const *mut T).cast_mut().cast::<c_void>(),
            (&columns as *const u64).cast_mut().cast(),
            (&scale as *const f32).cast_mut().cast(),
        ];
        let blocks = (heads * query_count) as u32;
        unsafe {
            driver::launch_kernel(
                softmax,
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                device.stream.cast(),
                &mut arguments,
            )
        }
        .expect("Failed to launch the softmax");
        outputs_gemm.launch(scores, values, outputs);
        device.synchronize();
    };
    launch(&mut scores, &mut outputs);
    let found: Vec<f64> = outputs.as_slice()[..expected.len()]
        .iter()
        .map(|value| value.to_f64())
        .collect();
    // The rival rounds its outputs to its input precision, coarser than the keys' summation error.
    let tolerance = GPU_TOLERANCE.max(if brain == 1 { 2.0 / 128.0 } else { 2.0 / 1024.0 });
    if let Err(error) = gpu_check(&format!("{name}/cublaslt"), &found, expected, width, false, tolerance) {
        println!("✗ {error}");
    }
    group.bench_function("cublaslt", |bench| bench.iter(|| launch(&mut scores, &mut outputs)));
    unsafe { driver::module::unload(module) }.expect("Failed to unload the softmax");
}

/// The group of `T` on each GPU: NumKong's bidirectional attention over random operands in
/// unified memory, each round synchronized and the first marked when it strays from the CPU.
#[cfg(feature = "cuda")]
fn bench_attention_gpu<T: GpuScalar + Attention>(
    c: &mut Criterion,
    settings: &Settings,
    devices: &[GpuDevice],
    dtype: &str,
    tolerance: f64,
    rival: Option<Rival<T>>,
) {
    let (key_count, depth, query_count) = (settings.dims_height, settings.dims_width, settings.dims_depth);
    let (width, seed) = (HEAD_COUNT * depth, u64::from(settings.seed));
    for device in devices {
        let name = format!("attention/{dtype}{}", device.label());
        if !settings.selects(&name) {
            continue;
        }
        let queries = device.random::<T>(&[query_count, width], seed);
        let keys = device.random::<T>(&[key_count, width], seed + 1);
        let values = device.random::<T>(&[key_count, width], seed + 2);
        let Some(key_values) = device.attention_pack(&keys, &values, depth) else {
            println!("{name} is unavailable: NumKong has no CUDA attention for {dtype}");
            continue;
        };
        let query_offsets = Tensor::<u32, _>::from_slice_in(&[0, query_count as u32], &[2], device.allocator())
            .expect("Failed to allocate");
        let mut outputs =
            Tensor::<f32, _>::zeros_in(&[query_count, width], device.allocator()).expect("Failed to allocate");
        let launch = |outputs: &mut Tensor<f32, UnifiedAllocator>| {
            device
                .attention(&queries, &key_values, &query_offsets, outputs)
                .expect("No CUDA attention");
            device.synchronize();
        };
        launch(&mut outputs);
        let operands = [&queries, &keys, &values];
        let expected = cpu_leading_rows(operands, depth);
        if let Err(error) = gpu_check(&name, outputs.as_slice(), &expected, width, false, tolerance) {
            println!("✗ {error}");
        }
        let mut group = c.benchmark_group(&name);
        group.throughput(Throughput::Elements(
            (4 * HEAD_COUNT * query_count * key_count * depth) as u64,
        ));
        group.bench_function("numkong", |bench| bench.iter(|| launch(&mut outputs)));
        if let Some(rival) = rival {
            rival(&mut group, &name, device, operands, depth, &expected);
        }
        group.finish();
    }
}

/// Every dtype NumKong's CUDA attention takes, on each GPU; cuBLASLt races the 16-bit floats.
#[cfg(feature = "cuda")]
fn bench_attention_gpus(c: &mut Criterion, settings: &Settings, devices: &[GpuDevice]) {
    bench_attention_gpu::<bf16>(
        c,
        settings,
        devices,
        "bf16",
        BF16_PROBABILITIES_TOLERANCE,
        Some(bench_cublaslt::<bf16>),
    );
    bench_attention_gpu::<f16>(c, settings, devices, "f16", GPU_TOLERANCE, Some(bench_cublaslt::<f16>));
    bench_attention_gpu::<e4m3>(c, settings, devices, "e4m3", BF16_PROBABILITIES_TOLERANCE, None);
    bench_attention_gpu::<i8>(c, settings, devices, "i8", GPU_TOLERANCE, None);
}

// endregion

// region: Main

fn main() {
    let settings = Settings::read("attention");
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    #[cfg(feature = "cuda")]
    let devices = gpu_devices(&settings);
    let mut criterion = numwars::configure_criterion(&settings);
    bench_attention(&mut criterion, &settings);
    #[cfg(feature = "cuda")]
    bench_attention_gpus(&mut criterion, &settings, &devices);
    criterion.final_summary();
}

// endregion
