# Reduce Benchmarks

Horizontal sum and row-norm benchmarks comparing NumKong against Polars, ndarray, and scalar baselines.

## Rust

| Library                     | Precision    |      GB/s |
| :-------------------------- | :----------- | --------: |
| ___Sum___                   |              |           |
| `numkong::reduce_moments`   | _u8 → u64_   | __33.61__ |
| `ndarray::sum`              | _f32 → f32_  | __32.50__ |
| `polars::ChunkedArray::sum` | _f64 → f64_  | __32.07__ |
| `polars::ChunkedArray::sum` | _f32 → f32_  |     31.26 |
| `ndarray::sum`              | _f64 → f64_  |     31.15 |
| `numkong::reduce_moments`   | _f32 → f64_  |     30.09 |
| `numkong::reduce_moments`   | _bf16 → f64_ |     25.45 |
| `numkong::reduce_moments`   | _f64 → f64_  |     22.51 |
| serial code                 | _u8 → u64_   |     12.51 |
| serial code                 | _f32 → f32_  |      6.38 |
| ___Row Norms___             |              |           |
| `ndarray::dot`              | _f64 → f64_  | __27.11__ |
| `numkong::Dot`              | _bf16 → f32_ | __24.46__ |
| `numkong::Dot`              | _f16 → f32_  | __23.18__ |
| `ndarray::dot`              | _f32 → f32_  |     21.63 |
| `numkong::Dot`              | _f64 → f64_  |     21.13 |
| `numkong::Dot`              | _f32 → f32_  |     20.01 |
| serial code                 | _f64 → f64_  |     12.77 |
| serial code                 | _f32 → f32_  |      6.54 |

## Python

| Library             | Precision     |      GB/s |
| :------------------ | :------------ | --------: |
| ___Sum___           |               |           |
| `numkong.sum`       | _u8 → u8_     | __33.51__ |
| `numkong.sum`       | _i8 → i8_     | __32.02__ |
| `numkong.sum`       | _f32 → f32_   | __29.17__ |
| `numkong.sum`       | _bf16 → bf16_ | __24.93__ |
| `numpy.sum`         | _f64 → f64_   | __24.16__ |
| `numkong.sum`       | _f64 → f64_   |     20.68 |
| `numpy.sum`         | _f32 → f32_   |     19.06 |
| `numpy.sum`         | _i8 → i8_     |      2.68 |
| `numpy.sum`         | _u8 → u8_     |      2.68 |
| ___Row Norms___     |               |           |
| `numkong.norm`      | _f32 → f64_   | __22.32__ |
| `numkong.norm`      | _f64 → f64_   | __18.17__ |
| `numkong.norm`      | _bf16 → f64_  | __17.82__ |
| `numpy.linalg.norm` | _f64 → f64_   |      8.21 |
| `numpy.linalg.norm` | _f32 → f64_   |      7.48 |

## Run It

### Rust

```bash
# Default 1M-element tensors
cargo bench --bench bench_reduce --features bench_reduce

# Smaller 10K-element tensors
NUMWARS_DIMS=10000 \
cargo bench --bench bench_reduce --features bench_reduce

# Focus on one operation
NUMWARS_FILTER="reduce/sum|reduce/row_norms" \
cargo bench --bench bench_reduce --features bench_reduce
```

### Python

```bash
python reduce/bench.py
```
