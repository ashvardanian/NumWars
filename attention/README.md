# Attention Benchmarks

Ragged scaled-dot-product attention over a pre-packed KV cache with 8 heads:

$$O = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

Speeds are in scalar operations per second: GSO/s on CPUs and TSO/s on GPUs.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, with the same input type; `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong's attention kernels are mixed-precision by design: _bf16_, _f16_, _e4m3_ and _i8_ inputs all produce _f32_ outputs.
The mainstream baselines have no fused CPU attention below _f32_ and _bf16_:

- __PyTorch__ (Python) ships a fused flash-style CPU `scaled_dot_product_attention`, routed through oneDNN on x86 and reference vector kernels on Arm.
- __candle__ (Rust) composes attention from `matmul` → `softmax_last_dim` → `matmul` over temporaries; its fused flash-attention crate is CUDA-only.
- __NumPy__ (Python) is the naive matmul → exp → normalize floor.

No baseline offers _e4m3_ or _i8_ attention on CPU, so those rows have no alternative.
The GPU rows race NumKong's CUDA kernels against:

- __PyTorch__ (Python) CUDA `scaled_dot_product_attention`, one row per backend that takes the shape: cuDNN, FlashAttention and the memory-efficient kernel.
- __cuBLASLt__ (Rust) in candle's composition: batched `Q × Kᵀ`, a softmax kernel, then batched `P × V`, all in the input precision.
- __CuPy__ (Python) running the NumPy floor in _f32_.

A `✗` marks a GPU row whose first round strays from NumKong's CPU attention by more than the rounding of its probability format.

## Intel Xeon5 Emerald Rapids

### Rust

| NumKong                                                   |      Speed | Speedup | Alternative                                |      Speed |
| :-------------------------------------------------------- | ---------: | ------: | :----------------------------------------- | ---------: |
| `numkong::AttentionPackedMatrix::attention`, _bf16 → f32_ |  719 GSO/s |       – | –                                          |          – |
| `numkong::AttentionPackedMatrix::attention`, _f16 → f32_  | 1.56 GSO/s |  0.049× | `candle_core::Tensor::matmul`, _f16 → f16_ | 31.7 GSO/s |
| `numkong::AttentionPackedMatrix::attention`, _e4m3 → f32_ |  718 GSO/s |       – | –                                          |          – |
| `numkong::AttentionPackedMatrix::attention`, _i8 → f32_   |  861 GSO/s |       – | –                                          |          – |

> Measured October 15, 2026.

### Python

Also benchmarked: `numpy.matmul`.

| NumKong                                  |      Speed | Speedup | Alternative                                                       |     Speed |
| :--------------------------------------- | ---------: | ------: | :---------------------------------------------------------------- | --------: |
| `numkong.attention_packed`, _bf16 → f32_ |  633 GSO/s |   0.94× | `torch.nn.functional.scaled_dot_product_attention`, _bf16 → bf16_ | 675 GSO/s |
| `numkong.attention_packed`, _f16 → f32_  | 1.57 GSO/s | 0.0090× | `torch.nn.functional.scaled_dot_product_attention`, _f16 → f16_   | 174 GSO/s |
| `numkong.attention_packed`, _e4m3 → f32_ |  614 GSO/s |       – | –                                                                 |         – |
| `numkong.attention_packed`, _i8 → f32_   |  861 GSO/s |       – | –                                                                 |         – |

> Measured October 15, 2026.

## 148× Nvidia SM100

### Rust

| NumKong                             |         Speed | Speedup | Alternative                     |     Speed |
| :---------------------------------- | ------------: | ------: | :------------------------------ | --------: |
| `nk_attention_packed`, _bf16 → f32_ |     912 TSO/s |   6.35× | `cublasLtMatmul`, _bf16 → bf16_ | 144 TSO/s |
| `nk_attention_packed`, _f16 → f32_  |     884 TSO/s |   6.03× | `cublasLtMatmul`, _f16 → f16_   | 147 TSO/s |
| `nk_attention_packed`, _e4m3 → f32_ | 1,011 TSO/s ✗ |       – | –                               |         – |
| `nk_attention_packed`, _i8 → f32_   |     864 TSO/s |       – | –                               |         – |

> Measured October 15, 2026.

### Python

Also benchmarked: `torch.nn.functional.scaled_dot_product_attention<flash>`, `torch.nn.functional.scaled_dot_product_attention<efficient>`, `cupy.matmul`.

| NumKong                                  |       Speed | Speedup | Alternative                                                              |       Speed |
| :--------------------------------------- | ----------: | ------: | :----------------------------------------------------------------------- | ----------: |
| `numkong.attention_packed`, _bf16 → f32_ |   911 TSO/s |   0.74× | `torch.nn.functional.scaled_dot_product_attention<cudnn>`, _bf16 → bf16_ | 1,224 TSO/s |
| `numkong.attention_packed`, _f16 → f32_  |   872 TSO/s |   0.73× | `torch.nn.functional.scaled_dot_product_attention<cudnn>`, _f16 → f16_   | 1,198 TSO/s |
| `numkong.attention_packed`, _e4m3 → f32_ | 981 TSO/s ✗ |       – | –                                                                        |           – |
| `numkong.attention_packed`, _i8 → f32_   |   826 TSO/s |       – | –                                                                        |           – |

> Measured October 15, 2026.

## Settings

| Variable              | Meaning            |
| :-------------------- | :----------------- |
| `NUMWARS_DIMS_HEIGHT` | KV context length  |
| `NUMWARS_DIMS_WIDTH`  | Head dimension     |
| `NUMWARS_DIMS_DEPTH`  | Query count        |
| `NUMWARS_FILTER`      | Rows to run        |
| `NUMWARS_THREADS`     | Competitor threads |

Throughput counts 4 · heads · queries · kv · head_dim scalar operations per call: two matrix products at 2 operations per multiply-add.

## Running

```bash
uv run --with numkong,numpy,tabulate,torch attention/bench.py
NUMWARS_DIMS_HEIGHT=4096 NUMWARS_DIMS_DEPTH=4096 uv run --with numkong,numpy,tabulate,torch attention/bench.py
cargo bench --features bench_attention --bench bench_attention
NUMWARS_FILTER="bf16|i8" cargo bench --features bench_attention

# Only the GPU rows, on device zero, 16384 queries over 16384 keys at head dimension 128
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda NUMWARS_SAMPLE_SIZE=10 \
NUMWARS_DIMS_HEIGHT=16384 NUMWARS_DIMS_WIDTH=128 NUMWARS_DIMS_DEPTH=16384 \
cargo bench --features "bench_attention cuda" --bench bench_attention
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda \
NUMWARS_DIMS_HEIGHT=16384 NUMWARS_DIMS_WIDTH=128 NUMWARS_DIMS_DEPTH=16384 python attention/bench.py
```
