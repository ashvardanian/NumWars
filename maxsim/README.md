# MaxSim Benchmarks

ColBERT-style late-interaction scoring benchmarks comparing NumKong against ndarray.

## Rust

| Library                              | Precision    |       GSO/s |
| :----------------------------------- | :----------- | ----------: |
| `numkong::MaxSimPackedMatrix::score` | _f32 → f64_  | __1483.41__ |
| `numkong::MaxSimPackedMatrix::score` | _bf16 → f32_ |  __983.57__ |
| `numkong::MaxSimPackedMatrix::score` | _f16 → f32_  |  __980.33__ |
| ndarray Q @ Dᵀ max-reduce            | _f32 → f32_  |       58.37 |

## Python

| Library                 | Precision    |       GSO/s |
| :---------------------- | :----------- | ----------: |
| `numkong.maxsim_packed` | _f32 → f64_  | __2425.72__ |
| `numpy` matmul          | _f32 → f32_  |     1525.56 |
| `numkong.maxsim_packed` | _bf16 → f32_ | __1236.30__ |
| `numkong.maxsim_packed` | _f16 → f32_  |  __696.78__ |

## Run It

### Rust

```bash
# Default 2048×2048×2048 workload
cargo bench --bench bench_maxsim --features bench_maxsim

# Smaller 128×128×256 workload
NUMWARS_DIMS_HEIGHT=128 NUMWARS_DIMS_WIDTH=128 NUMWARS_DIMS_DEPTH=256 \
cargo bench --bench bench_maxsim --features bench_maxsim

# Focus on one dtype
NUMWARS_FILTER="maxsim/f32" \
cargo bench --bench bench_maxsim --features bench_maxsim
```

### Python

```bash
uv run --with numkong,numpy,tabulate,ml_dtypes python maxsim/bench.py
```
