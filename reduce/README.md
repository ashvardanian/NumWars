# Reduce Benchmarks

Horizontal sum and row-norm benchmarks comparing NumKong against Polars, ndarray, and scalar baselines.

## Rust

| Library                     | Precision    |       GB/s |
| :-------------------------- | :----------- | ---------: |
| ___Sum___                   |              |            |
| `polars::ChunkedArray::sum` | _f64 → f64_  | __105.77__ |
| `polars::ChunkedArray::sum` | _f32 → f32_  | __103.10__ |
| `ndarray::sum`              | _f64 → f64_  |      92.66 |
| `ndarray::sum`              | _f32 → f32_  |      46.41 |
| `numkong::reduce_moments`   | _bf16 → f64_ |  __30.89__ |
| `numkong::reduce_moments`   | _u8 → u64_   |  __22.58__ |
| serial code                 | _u8 → u64_   |      21.38 |
| `numkong::reduce_moments`   | _f64 → f64_  |      17.01 |
| `numkong::reduce_moments`   | _f32 → f64_  |  __9.602__ |
| serial code                 | _f32 → f32_  |       7.92 |
| ___Row Norms___             |              |            |
| `ndarray::dot`              | _f64 → f64_  |  __83.56__ |
| `ndarray::dot`              | _f32 → f32_  |  __49.58__ |
| `numkong::Dot`              | _bf16 → f32_ |  __28.54__ |
| `numkong::Dot`              | _f64 → f64_  |      21.83 |
| serial code                 | _f64 → f64_  |      16.72 |
| `numkong::Dot`              | _f16 → f32_  |  __12.04__ |
| `numkong::Dot`              | _f32 → f32_  |      9.872 |
| serial code                 | _f32 → f32_  |       8.57 |

## Python

| Library             | Precision   |      GB/s |
| :------------------ | :---------- | --------: |
| ___Sum___           |             |           |
| `numpy.sum`         | _f64 → f64_ |     57.05 |
| `numpy.sum`         | _f32 → f32_ |     31.59 |
| `numkong.sum`       | _u8 → u8_   | __20.28__ |
| `numkong.sum`       | _i8 → i8_   | __19.93__ |
| `numkong.sum`       | _f64 → f64_ | __15.22__ |
| `numkong.sum`       | _f32 → f32_ |  __8.84__ |
| `numpy.sum`         | _u8 → u8_   |      6.53 |
| `numpy.sum`         | _i8 → i8_   |      6.27 |
| ___Norm___          |             |           |
| `numpy.linalg.norm` | _f64 → f64_ |     28.18 |
| `numpy.linalg.norm` | _f32 → f64_ |     18.77 |
| `numkong.norm`      | _f64 → f64_ | __16.24__ |
| `numkong.norm`      | _f32 → f64_ | __14.06__ |

## Run It

### Rust

```bash
# Default 2048-element tensors
cargo bench --bench bench_reduce --features bench_reduce

# Larger 1M-element tensors
NUMWARS_BATCH_PER_CORE=1000000 \
cargo bench --bench bench_reduce --features bench_reduce

# Focus on one operation
NUMWARS_FILTER="reduce/sum|reduce/row_norms" \
cargo bench --bench bench_reduce --features bench_reduce
```

### Python

```bash
python reduce/bench.py
```
