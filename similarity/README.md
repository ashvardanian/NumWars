# Similarity Benchmarks

Pairwise vector-vector distance and dot-product benchmarks comparing NumKong against scalar baselines and BLAS-backed libraries.

## Rust

| Library                    | Precision    |     GSO/s |
| :------------------------- | :----------- | --------: |
| ___Dot___                  |              |           |
| `numkong::Dot`             | _u8 → u32_   | __54.28__ |
| `numkong::Dot`             | _i8 → i32_   | __43.18__ |
| `numkong::Dot`             | _bf16 → f32_ | __20.09__ |
| serial code                | _f32 → f32_  |     14.25 |
| `ndarray::dot`             | _f32 → f32_  |      7.75 |
| `nalgebra::dot`            | _f32 → f32_  |      7.56 |
| serial code                | _u8 → u32_   |      7.20 |
| `numkong::Dot`             | _f32 → f64_  |  __6.12__ |
| serial code                | _i8 → i32_   |      4.73 |
| serial code                | _bf16 → f32_ |      0.16 |
| `numkong::Dot`             | _f16 → f32_  |         ? |
| `numkong::Dot`             | _f64 → f64_  |         ? |
| `numkong::Dot`             | _e4m3 → f32_ |         ? |
| `numkong::Dot`             | _e5m2 → f32_ |         ? |
| `numkong::Dot`             | _e2m3 → f32_ |         ? |
| `numkong::Dot`             | _e3m2 → f32_ |         ? |
| `ndarray::dot`             | _f64 → f64_  |         ? |
| `nalgebra::dot`            | _f64 → f64_  |         ? |
| ___Angular___              |              |           |
| `numkong::Angular`         | _f32 → f32_  |         ? |
| `numkong::Angular`         | _f64 → f64_  |         ? |
| `numkong::Angular`         | _i8 → f32_   |         ? |
| `numkong::Angular`         | _u8 → f32_   |         ? |
| `numkong::Angular`         | _f16 → f32_  |         ? |
| `numkong::Angular`         | _bf16 → f32_ |         ? |
| `numkong::Angular`         | _e4m3 → f32_ |         ? |
| `numkong::Angular`         | _e5m2 → f32_ |         ? |
| `numkong::Angular`         | _e2m3 → f32_ |         ? |
| `numkong::Angular`         | _e3m2 → f32_ |         ? |
| ___Euclidean___            |              |           |
| `numkong::Euclidean`       | _u8 → f32_   | __40.83__ |
| `numkong::Euclidean`       | _i8 → f32_   | __34.10__ |
| `numkong::Euclidean`       | _bf16 → f32_ | __12.65__ |
| `numkong::Euclidean`       | _f32 → f64_  |  __5.53__ |
| `ndarray::norm`            | _f32 → f32_  |      4.75 |
| `nalgebra::norm`           | _f32 → f32_  |      4.63 |
| serial code                | _f32 → f32_  |      1.62 |
| serial code                | _u8 → f32_   |      1.18 |
| serial code                | _i8 → f32_   |      1.17 |
| serial code                | _bf16 → f32_ |      0.16 |
| `numkong::Euclidean`       | _f16 → f32_  |         ? |
| `numkong::Euclidean`       | _f64 → f64_  |         ? |
| `numkong::Euclidean`       | _e4m3 → f32_ |         ? |
| `numkong::Euclidean`       | _e5m2 → f32_ |         ? |
| `numkong::Euclidean`       | _e2m3 → f32_ |         ? |
| `numkong::Euclidean`       | _e3m2 → f32_ |         ? |
| `ndarray::norm`            | _f64 → f64_  |         ? |
| `nalgebra::norm`           | _f64 → f64_  |         ? |
| ___Squared Euclidean___    |              |           |
| `numkong::SqEuclidean`     | _f32 → f32_  |         ? |
| `numkong::SqEuclidean`     | _f64 → f64_  |         ? |
| `numkong::SqEuclidean`     | _i8 → u32_   |         ? |
| `numkong::SqEuclidean`     | _u8 → u32_   |         ? |
| `numkong::SqEuclidean`     | _f16 → f32_  |         ? |
| `numkong::SqEuclidean`     | _bf16 → f32_ |         ? |
| `numkong::SqEuclidean`     | _e4m3 → f32_ |         ? |
| `numkong::SqEuclidean`     | _e5m2 → f32_ |         ? |
| `numkong::SqEuclidean`     | _e2m3 → f32_ |         ? |
| `numkong::SqEuclidean`     | _e3m2 → f32_ |         ? |
| ___Hamming___              |              |           |
| `numkong::Hamming`         | _u1x8 → u32_ |         ? |
| ___Jaccard___              |              |           |
| `numkong::Jaccard`         | _u1x8 → f32_ |         ? |
| ___Kullback-Leibler___     |              |           |
| `numkong::KullbackLeibler` | _f32 → f32_  |         ? |
| `numkong::KullbackLeibler` | _f64 → f64_  |         ? |
| ___Jensen-Shannon___       |              |           |
| `numkong::JensenShannon`   | _f32 → f32_  |         ? |
| `numkong::JensenShannon`   | _f64 → f64_  |         ? |

## Python

| Library             | Precision    |    GSO/s |
| :------------------ | :----------- | -------: |
| ___Dot___           |              |          |
| `numkong.dot`       | _u8 → u32_   | __4.88__ |
| `numkong.dot`       | _i8 → i32_   | __3.25__ |
| `scipy.blas.sdot`   | _f32 → f32_  |     3.14 |
| `numkong.dot`       | _f32 → f64_  | __2.76__ |
| `numkong.dot`       | _bf16 → f32_ | __0.37__ |
| ___Euclidean___     |              |          |
| `numkong.euclidean` | _u8 → f32_   | __5.65__ |
| `numkong.euclidean` | _i8 → f32_   | __5.08__ |
| `numkong.euclidean` | _f32 → f64_  | __3.33__ |
| `scipy.euclidean`   | _u8 → f32_   |     0.48 |
| `numkong.euclidean` | _bf16 → f32_ | __0.41__ |
| `scipy.euclidean`   | _i8 → f32_   |     0.38 |
| `scipy.euclidean`   | _f32 → f32_  |     0.38 |

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
