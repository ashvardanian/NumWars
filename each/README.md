# Each Benchmarks

Elementwise FMA (fused multiply-add) bandwidth benchmarks comparing NumKong against ndarray and NumPy.

## Rust

| Library        | Precision     |       GB/s |
| :------------- | :------------ | ---------: |
| `numkong::fma` | _f32 → f32_   | __121.56__ |
| `scalar fma`   | _f32 → f32_   |     112.96 |
| `ndarray fma`  | _f32 → f32_   |      42.30 |
| `numkong::fma` | _f16 → f16_   |  __22.51__ |
| `numkong::fma` | _bf16 → bf16_ |  __19.47__ |

## Python

| Library       | Precision     |     GB/s |
| :------------ | :------------ | -------: |
| `numkong fma` | _f32 → f32_   | __1.24__ |
| `numkong fma` | _f16 → f16_   | __0.59__ |
| `numpy fma`   | _f32 → f32_   |     0.30 |
| `numkong fma` | _bf16 → bf16_ | __0.06__ |
| `numpy fma`   | _f16 → f16_   |     0.01 |

## Run It

### Rust

```bash
# Default 1M-element tensors
cargo bench --bench bench_each --features bench_each

# Focus on one operation family
NUMWARS_FILTER="each/sum|each/fma" \
cargo bench --bench bench_each --features bench_each
```

### Python

```bash
# Default 1M-element tensors, add on float32
python each/bench.py --filter 'add/float32'

# Default 1M-element tensors, multiply on float32
python each/bench.py --filter 'multiply/float32'
```
