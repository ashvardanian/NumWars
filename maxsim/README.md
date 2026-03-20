# MaxSim Benchmarks

ColBERT-style late-interaction scoring benchmarks comparing NumKong against ndarray.

## Rust

| Library                       | Precision    |      GSO/s |
| :---------------------------- | :----------- | ---------: |
| `numkong::MaxSimPackedMatrix` | _f16 → f32_  | __423.69__ |
| `numkong::MaxSimPackedMatrix` | _f32 → f64_  | __415.47__ |
| `numkong::MaxSimPackedMatrix` | _bf16 → f32_ | __224.48__ |
| `ndarray::dot`                | _f32 → f32_  |      38.36 |

## Python

| Library                 | Precision    |      GSO/s |
| :---------------------- | :----------- | ---------: |
| `numkong.maxsim_packed` | _f16 → f32_  | __833.26__ |
| `numkong.maxsim_packed` | _f32 → f64_  | __776.43__ |
| `numkong.maxsim_packed` | _bf16 → f32_ | __428.56__ |
| `numpy` matmul          | _f32 → f32_  |     129.03 |

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
