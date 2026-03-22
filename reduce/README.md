# Reduce Benchmarks

Horizontal sum and row-norm benchmarks comparing NumKong against Polars, ndarray, and scalar baselines.

## Rust

| Library                     | Precision    |       GB/s |
| :-------------------------- | :----------- | ---------: |
| ___Sum___                   |              |            |
| `polars::ChunkedArray::sum` | _f64 → f64_  | __113.57__ |
| `polars::ChunkedArray::sum` | _f32 → f32_  | __110.70__ |
| `ndarray::sum`              | _f64 → f64_  |     99.49 |
| `ndarray::sum`              | _f32 → f32_  |     49.83 |
| `numkong::reduce_moments`   | _bf16 → f64_ | __33.17__ |
| `numkong::reduce_moments`   | _u8 → u64_   | __24.24__ |
| serial code                 | _u8 → u64_   |     22.96 |
| `numkong::reduce_moments`   | _f64 → f64_  |     18.26 |
| `numkong::reduce_moments`   | _f32 → f64_  | __10.31__ |
| serial code                 | _f32 → f32_  |      8.50 |
| ___Row Norms___             |              |            |
| `ndarray::dot`              | _f64 → f64_  |  __89.72__ |
| `ndarray::dot`              | _f32 → f32_  |  __53.24__ |
| `numkong::Dot`              | _bf16 → f32_ |  __30.64__ |
| `numkong::Dot`              | _f64 → f64_  |     23.44 |
| serial code                 | _f64 → f64_  |     17.95 |
| `numkong::Dot`              | _f16 → f32_  |  __12.93__ |
| `numkong::Dot`              | _f32 → f32_  |     10.60 |
| serial code                 | _f32 → f32_  |      9.20 |

## Python

| Library             | Precision   |      GB/s |
| :------------------ | :---------- | --------: |
| ___Sum___           |             |           |
| `numpy.sum`         | _f64 → f64_ |     61.26 |
| `numpy.sum`         | _f32 → f32_ |     33.92 |
| `numkong.sum`       | _u8 → u8_   | __21.78__ |
| `numkong.sum`       | _i8 → i8_   | __21.40__ |
| `numkong.sum`       | _f64 → f64_ | __16.34__ |
| `numkong.sum`       | _f32 → f32_ |  __9.49__ |
| `numpy.sum`         | _u8 → u8_   |      7.01 |
| `numpy.sum`         | _i8 → i8_   |      6.73 |
| ___Norm___          |             |           |
| `numpy.linalg.norm` | _f64 → f64_ |     30.26 |
| `numpy.linalg.norm` | _f32 → f64_ |     20.15 |
| `numkong.norm`      | _f64 → f64_ | __17.44__ |
| `numkong.norm`      | _f32 → f64_ | __15.10__ |

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
