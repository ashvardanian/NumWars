# Attention Benchmarks

Ragged scaled-dot-product attention over a pre-packed KV cache with 8 heads:

$$O = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

NumKong's attention kernels are mixed-precision by design: _bf16_, _e4m3_ and _i8_ inputs all produce _f32_ outputs.
The mainstream baselines have no fused CPU attention below _f32_ and _bf16_:

- __PyTorch__ (Python) ships a fused flash-style CPU `scaled_dot_product_attention`, routed through oneDNN on x86 and reference vector kernels on Arm.
- __candle__ (Rust) composes attention from `matmul` → `softmax_last_dim` → `matmul` over temporaries; its fused flash-attention crate is CUDA-only.
- __NumPy__ (Python) is the naive matmul → exp → normalize floor.

No baseline offers _e4m3_ or _i8_ attention on CPU, so those rows compare NumKong with the strongest higher-precision competitor.

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
```
