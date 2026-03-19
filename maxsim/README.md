# MaxSim Benchmarks

ColBERT-style late-interaction scoring benchmarks comparing NumKong against ndarray.

## Rust

| Library                       | Precision    |      GSO/s |
| :---------------------------- | :----------- | ---------: |
| `numkong::MaxSimPackedMatrix` | _f16 → f32_  | __423.69__ |
| `numkong::MaxSimPackedMatrix` | _f32 → f64_  | __415.47__ |
| `numkong::MaxSimPackedMatrix` | _bf16 → f32_ | __224.48__ |
| `ndarray::dot`                | _f32 → f32_  |      38.36 |

## Run It

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
