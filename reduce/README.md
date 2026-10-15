# Reduce Benchmarks

Horizontal sum, min-max and row-norm benchmarks comparing NumKong against Polars, ndarray, argminmax, NumPy, and scalar baselines.
The min-max rows find the smallest and the largest element: NumKong returns both values with their indices, argminmax, NumPy and the serial loop return the indices, and Polars returns the values.
The unit is GB/s.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong widens its sums: u8 → u64, bf16 → f32, f32 → f64, and f32 and bf16 norms → f64.
The `baseline_*` and `bench_row_norms_mini` alternatives are the hand-written serial loops of `bench.rs`, and `RunBaseline::run` is its inlined u8 sum.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `polars::ChunkedArray::sum`, `baseline_sum`, `baseline_minmax`, `baseline_row_norms_float`.

| NumKong                                                |     Speed | Speedup | Alternative                     |     Speed |
| :----------------------------------------------------- | --------: | ------: | :------------------------------ | --------: |
| `numkong::ReduceMoments::reduce_moments`, _bf16 → f32_ | 33.4 GB/s |       – | –                               |         – |
| `numkong::ReduceMoments::reduce_moments`, _u8 → u64_   | 45.9 GB/s |   3.14× | `RunBaseline::run`              | 14.6 GB/s |
| `numkong::ReduceMoments::reduce_moments`, _f64 → f64_  | 28.9 GB/s |   0.28× | `ndarray::sum`                  |  103 GB/s |
| `numkong::ReduceMoments::reduce_moments`, _f32 → f64_  | 50.7 GB/s |   0.78× | `ndarray::sum`, _f32 → f32_     | 64.7 GB/s |
| `numkong::ReduceMinMax::reduce_minmax`, _f32 → f32_    | 47.8 GB/s |   1.68× | `polars::ChunkedArray::min/max` | 28.4 GB/s |
| `numkong::ReduceMinMax::reduce_minmax`, _f64 → f64_    | 52.2 GB/s |   1.70× | `polars::ChunkedArray::min/max` | 30.7 GB/s |
| `numkong::ReduceMinMax::reduce_minmax`, _u8 → u8_      | 47.0 GB/s |   3.67× | `argminmax::ArgMinMax`          | 12.8 GB/s |
| `numkong::ReduceMinMax::reduce_minmax`, _bf16 → bf16_  | 31.0 GB/s |       – | –                               |         – |
| `numkong::Dot::dot`, _bf16 → f32_                      | 24.6 GB/s |   47.3× | `bench_row_norms_mini`          | 0.52 GB/s |
| `numkong::Dot::dot`, _f64 → f64_                       | 19.6 GB/s |   0.75× | `ndarray::dot`                  | 26.2 GB/s |
| `numkong::Dot::dot`, _f16 → f32_                       | 21.7 GB/s |   41.7× | `bench_row_norms_mini`          | 0.52 GB/s |
| `numkong::Dot::dot`, _f32 → f64_                       | 18.1 GB/s |   0.73× | `ndarray::dot`, _f32 → f32_     | 24.8 GB/s |

> Measured October 14, 2026.

### Python

| NumKong                         |     Speed | Speedup | Alternative                   |     Speed |
| :------------------------------ | --------: | ------: | :---------------------------- | --------: |
| `numkong.sum`, _u8 → u8_        | 12.1 GB/s |   14.3× | `numpy.sum`                   | 0.85 GB/s |
| `numkong.sum`, _i8 → i8_        | 11.9 GB/s |   14.0× | `numpy.sum`                   | 0.85 GB/s |
| `numkong.sum`, _f64 → f64_      | 23.9 GB/s |   2.35× | `numpy.sum`                   | 10.2 GB/s |
| `numkong.sum`, _f32 → f32_      | 32.1 GB/s |   6.26× | `numpy.sum`                   | 5.12 GB/s |
| `numkong.sum`, _bf16 → bf16_    | 16.7 GB/s |       – | –                             |         – |
| `numkong.minmax`, _f32 → f32_   | 22.2 GB/s |   5.17× | `numpy.argmin → numpy.argmax` | 4.29 GB/s |
| `numkong.minmax`, _f64 → f64_   | 30.7 GB/s |   3.93× | `numpy.argmin → numpy.argmax` | 7.81 GB/s |
| `numkong.minmax`, _u8 → u8_     | 11.2 GB/s |   10.1× | `numpy.argmin → numpy.argmax` | 1.10 GB/s |
| `numkong.minmax`, _bf16 → bf16_ | 10.6 GB/s |       – | –                             |         – |
| `numkong.norm`, _f64 → f64_     | 19.3 GB/s |   2.71× | `numpy.linalg.norm`           | 7.12 GB/s |
| `numkong.norm`, _f32 → f64_     | 23.7 GB/s |   3.28× | `numpy.linalg.norm`           | 7.23 GB/s |
| `numkong.norm`, _bf16 → f64_    | 18.7 GB/s |       – | –                             |         – |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `ndarray::sum`, `baseline_sum`, `baseline_row_norms_float`.

| NumKong                                                |     Speed | Speedup | Alternative                              |     Speed |
| :----------------------------------------------------- | --------: | ------: | :--------------------------------------- | --------: |
| `numkong::ReduceMoments::reduce_moments`, _bf16 → f32_ | 30.9 GB/s |       – | –                                        |         – |
| `numkong::ReduceMoments::reduce_moments`, _u8 → u64_   | 22.6 GB/s |   1.06× | `RunBaseline::run`                       | 21.4 GB/s |
| `numkong::ReduceMoments::reduce_moments`, _f64 → f64_  | 17.0 GB/s |   0.16× | `polars::ChunkedArray::sum`              |  106 GB/s |
| `numkong::ReduceMoments::reduce_moments`, _f32 → f64_  | 9.60 GB/s |  0.093× | `polars::ChunkedArray::sum`, _f32 → f32_ |  103 GB/s |
| `numkong::Dot::dot`, _bf16 → f32_                      | 28.5 GB/s |       – | `bench_row_norms_mini`                   |         ⋯ |
| `numkong::Dot::dot`, _f64 → f64_                       | 21.8 GB/s |   0.26× | `ndarray::dot`                           | 83.6 GB/s |
| `numkong::Dot::dot`, _f16 → f32_                       | 12.0 GB/s |       – | `bench_row_norms_mini`                   |         ⋯ |
| `numkong::Dot::dot`, _f32 → f64_                       | 9.87 GB/s |   0.20× | `ndarray::dot`, _f32 → f32_              | 49.6 GB/s |

> Measured October 1, 2026.

### Python

| NumKong                     |     Speed | Speedup | Alternative         |     Speed |
| :-------------------------- | --------: | ------: | :------------------ | --------: |
| `numkong.sum`, _u8 → u8_    | 20.3 GB/s |   3.11× | `numpy.sum`         | 6.53 GB/s |
| `numkong.sum`, _i8 → i8_    | 19.9 GB/s |   3.18× | `numpy.sum`         | 6.27 GB/s |
| `numkong.sum`, _f64 → f64_  | 15.2 GB/s |   0.27× | `numpy.sum`         | 57.1 GB/s |
| `numkong.sum`, _f32 → f32_  | 8.84 GB/s |   0.28× | `numpy.sum`         | 31.6 GB/s |
| `numkong.norm`, _f64 → f64_ | 16.2 GB/s |   0.58× | `numpy.linalg.norm` | 28.2 GB/s |
| `numkong.norm`, _f32 → f64_ | 14.1 GB/s |   0.75× | `numpy.linalg.norm` | 18.8 GB/s |

> Measured October 1, 2026.

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
