# Similarity Benchmarks

Pairwise vector-vector distance and dot-product benchmarks comparing NumKong against scalar baselines and BLAS-backed libraries.

## Rust

| Library               | Precision    |     GSO/s |
| :-------------------- | :----------- | --------: |
| `numkong::dot`        | _u8 → u32_   | __54.28__ |
| `numkong::dot`        | _i8 → i32_   | __43.18__ |
| `numkong::euclidean`  | _u8 → f32_   | __40.83__ |
| `numkong::euclidean`  | _i8 → f32_   | __34.10__ |
| `numkong::dot`        | _bf16 → f32_ | __20.09__ |
| `scalar dot`          | _f32 → f32_  |     14.25 |
| `numkong::euclidean`  | _bf16 → f32_ | __12.65__ |
| `ndarray::dot`        | _f32 → f32_  |      7.75 |
| `nalgebra::dot`       | _f32 → f32_  |      7.56 |
| `scalar dot`          | _u8 → u32_   |      7.20 |
| `numkong::dot`        | _f32 → f64_  |  __6.12__ |
| `numkong::euclidean`  | _f32 → f64_  |  __5.53__ |
| `ndarray::euclidean`  | _f32 → f32_  |      4.75 |
| `scalar dot`          | _i8 → i32_   |      4.73 |
| `nalgebra::euclidean` | _f32 → f32_  |      4.63 |
| `scalar euclidean`    | _f32 → f32_  |      1.62 |
| `scalar euclidean`    | _u8 → f32_   |      1.18 |
| `scalar euclidean`    | _i8 → f32_   |      1.17 |
| `scalar euclidean`    | _bf16 → f32_ |      0.16 |
| `scalar dot`          | _bf16 → f32_ |      0.16 |

## Python

| Library              | Precision    |    GSO/s |
| :------------------- | :----------- | -------: |
| `numkong::euclidean` | _u8 → f32_   | __5.65__ |
| `numkong::euclidean` | _i8 → f32_   | __5.08__ |
| `numkong::dot`       | _u8 → u32_   | __4.88__ |
| `numkong::euclidean` | _f32 → f64_  | __3.33__ |
| `numkong::dot`       | _i8 → i32_   | __3.25__ |
| `scipy sdot`         | _f32 → f32_  |     3.14 |
| `numkong::dot`       | _f32 → f64_  | __2.76__ |
| `scipy euclidean`    | _u8 → f32_   |     0.48 |
| `numkong::euclidean` | _bf16 → f32_ | __0.41__ |
| `scipy euclidean`    | _i8 → f32_   |     0.38 |
| `scipy euclidean`    | _f32 → f32_  |     0.38 |
| `numkong::dot`       | _bf16 → f32_ | __0.37__ |

## Run It

### Rust

```bash
# Default 2048-dimensional vectors
cargo bench --bench bench_similarity --features bench_similarity

# Smaller 512-dimensional vectors
NUMWARS_DIMS=512 cargo bench --bench bench_similarity --features bench_similarity

# One benchmark group
NUMWARS_FILTER="similarity/angular/f32" \
cargo bench --bench bench_similarity --features bench_similarity
```

### Python

```bash
# Default 2048-dimensional pairwise distances
python similarity/bench.py --filter 'angular.*float32'

# Compare probability metrics
python similarity/bench.py --filter 'jensenshannon|kullback' --ndim 1536 --count 128
```
