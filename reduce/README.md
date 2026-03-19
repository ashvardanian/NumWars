# Reduce Benchmarks

Horizontal sum and row-norm benchmarks comparing NumKong against Polars, ndarray, and scalar baselines.

## Rust

| Library                     | Precision    |      GB/s |
| :-------------------------- | :----------- | --------: |
| ___Sum___                   |              |           |
| `polars::ChunkedArray::sum` | _f32 → f32_  | __43.86__ |
| `numkong::reduce_moments`   | _f32 → f64_  |     43.26 |
| `ndarray::sum`              | _f32 → f32_  |     36.38 |
| serial code                 | _f32 → f32_  |      6.67 |
| `numkong::reduce_moments`   | _f64 → f64_  |         ? |
| `polars::ChunkedArray::sum` | _f64 → f64_  |         ? |
| `ndarray::sum`              | _f64 → f64_  |         ? |
| ___Row Norms___             |              |           |
| `numkong::Dot`              | _f32 → f32_  |         ? |
| serial code                 | _f32 → f32_  |         ? |
| `ndarray::dot`              | _f32 → f32_  |         ? |
| `numkong::Dot`              | _f64 → f64_  |         ? |
| serial code                 | _f64 → f64_  |         ? |
| `ndarray::dot`              | _f64 → f64_  |         ? |
| `numkong::Dot`              | _bf16 → f32_ |         ? |
| `numkong::Dot`              | _f16 → f32_  |         ? |

## Python

| Library             | Precision   |      GB/s |
| :------------------ | :---------- | --------: |
| ___Sum___           |             |           |
| `numpy.sum`         | _f64 → f64_ | __24.75__ |
| `numpy.sum`         | _f32 → f32_ |     18.01 |
| `numpy.sum`         | _i8 → i8_   |      2.81 |
| ___Row Norms___     |             |           |
| `numpy.linalg.norm` | _f64 → f64_ |      7.52 |
| `numpy.linalg.norm` | _f32 → f64_ |      6.49 |

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
