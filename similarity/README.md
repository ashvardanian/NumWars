# Similarity Benchmarks

Pairwise vector-vector distance and dot-product benchmarks comparing NumKong against scalar baselines and BLAS-backed libraries.

## Rust

| Library                    | Precision    |     GSO/s |
| :------------------------- | :----------- | --------: |
| ___Dot___                  |              |           |
| `numkong::Dot`             | _u8 → u32_   | __37.34__ |
| `numkong::Dot`             | _i8 → i32_   | __37.41__ |
| `numkong::Dot`             | _bf16 → f32_ | __14.83__ |
| serial code                | _f32 → f32_  | __13.86__ |
| `ndarray::dot`             | _f32 → f32_  |     12.77 |
| `nalgebra::dot`            | _f32 → f32_  |     12.95 |
| serial code                | _u8 → u32_   |     19.28 |
| `numkong::Dot`             | _f32 → f64_  |  __2.50__ |
| serial code                | _i8 → i32_   |     19.70 |
| serial code                | _bf16 → f32_ |      0.67 |
| `numkong::Dot`             | _f16 → f32_  |  __6.16__ |
| `numkong::Dot`             | _f64 → f64_  |      2.81 |
| `numkong::Dot`             | _e4m3 → f32_ |  __4.43__ |
| `numkong::Dot`             | _e5m2 → f32_ |  __6.08__ |
| `numkong::Dot`             | _e2m3 → f32_ | __26.64__ |
| `numkong::Dot`             | _e3m2 → f32_ | __11.32__ |
| `ndarray::dot`             | _f64 → f64_  |     11.62 |
| `nalgebra::dot`            | _f64 → f64_  | __11.66__ |
| ___Angular___              |              |           |
| `numkong::Angular`         | _f32 → f32_  |  __2.34__ |
| `numkong::Angular`         | _f64 → f64_  |  __2.30__ |
| `numkong::Angular`         | _i8 → f32_   | __31.86__ |
| `numkong::Angular`         | _u8 → f32_   | __31.86__ |
| `numkong::Angular`         | _f16 → f32_  |  __4.81__ |
| `numkong::Angular`         | _bf16 → f32_ | __10.28__ |
| `numkong::Angular`         | _e4m3 → f32_ |  __2.19__ |
| `numkong::Angular`         | _e5m2 → f32_ |  __5.34__ |
| `numkong::Angular`         | _e2m3 → f32_ |  __2.60__ |
| `numkong::Angular`         | _e3m2 → f32_ |  __2.60__ |
| ___Euclidean___            |              |           |
| `numkong::Euclidean`       | _u8 → f32_   | __34.85__ |
| `numkong::Euclidean`       | _i8 → f32_   | __34.91__ |
| `numkong::Euclidean`       | _bf16 → f32_ |  __5.54__ |
| `numkong::Euclidean`       | _f32 → f64_  |  __2.50__ |
| `ndarray::norm`            | _f32 → f32_  |      7.61 |
| `nalgebra::norm`           | _f32 → f32_  |      7.74 |
| serial code                | _f32 → f32_  | __11.15__ |
| serial code                | _u8 → f32_   |      9.19 |
| serial code                | _i8 → f32_   |      9.26 |
| serial code                | _bf16 → f32_ |      0.66 |
| `numkong::Euclidean`       | _f16 → f32_  |  __5.27__ |
| `numkong::Euclidean`       | _f64 → f64_  |      2.60 |
| `numkong::Euclidean`       | _e4m3 → f32_ |  __2.28__ |
| `numkong::Euclidean`       | _e5m2 → f32_ |  __5.80__ |
| `numkong::Euclidean`       | _e2m3 → f32_ |  __2.74__ |
| `numkong::Euclidean`       | _e3m2 → f32_ |  __2.75__ |
| `ndarray::norm`            | _f64 → f64_  |      5.59 |
| `nalgebra::norm`           | _f64 → f64_  |  __5.63__ |
| ___Squared Euclidean___    |              |           |
| `numkong::SqEuclidean`     | _f32 → f32_  |  __2.65__ |
| `numkong::SqEuclidean`     | _f64 → f64_  |  __2.76__ |
| `numkong::SqEuclidean`     | _i8 → f32_   | __38.59__ |
| `numkong::SqEuclidean`     | _u8 → f32_   | __38.80__ |
| `numkong::SqEuclidean`     | _f16 → f32_  |  __5.52__ |
| `numkong::SqEuclidean`     | _bf16 → f32_ |  __5.70__ |
| `numkong::SqEuclidean`     | _e4m3 → f32_ |  __2.29__ |
| `numkong::SqEuclidean`     | _e5m2 → f32_ |  __5.87__ |
| `numkong::SqEuclidean`     | _e2m3 → f32_ |  __2.76__ |
| `numkong::SqEuclidean`     | _e3m2 → f32_ |  __2.76__ |
| ___Hamming___              |              |           |
| `numkong::Hamming`         | _u1x8 → u32_ | __45.20__ |
| ___Jaccard___              |              |           |
| `numkong::Jaccard`         | _u1x8 → f32_ | __36.88__ |
| ___Kullback-Leibler___     |              |           |
| `numkong::KullbackLeibler` | _f32 → f32_  |  __2.87__ |
| `numkong::KullbackLeibler` | _f64 → f64_  |  __0.27__ |
| ___Jensen-Shannon___       |              |           |
| `numkong::JensenShannon`   | _f32 → f32_  |  __1.47__ |
| `numkong::JensenShannon`   | _f64 → f64_  |  __0.17__ |

## Python

| Library                | Precision    |     GSO/s |
| :--------------------- | :----------- | --------: |
| ___Dot___              |              |           |
| `numkong.dot`          | _u8 → f32_   |  __9.37__ |
| `numkong.dot`          | _i8 → f32_   |  __9.81__ |
| `numkong.dot`          | _bf16 → f32_ |  __1.06__ |
| `numkong.dot`          | _f32 → f32_  |  __2.02__ |
| `numkong.dot`          | _f64 → f32_  |  __2.48__ |
| `numpy.dot`            | _f32 → f32_  |      3.81 |
| `numpy.dot`            | _f64 → f64_  |      3.68 |
| ___Angular___          |              |           |
| `numkong.angular`      | _u8 → f32_   | __10.07__ |
| `numkong.angular`      | _i8 → f32_   | __10.04__ |
| `numkong.angular`      | _bf16 → f32_ |  __1.03__ |
| `numkong.angular`      | _f32 → f32_  |  __1.98__ |
| `numkong.angular`      | _f64 → f32_  |  __2.01__ |
| ___Euclidean___        |              |           |
| `numkong.euclidean`    | _u8 → f32_   | __10.24__ |
| `numkong.euclidean`    | _i8 → f32_   | __10.32__ |
| `numkong.euclidean`    | _bf16 → f32_ |  __0.92__ |
| `numkong.euclidean`    | _f32 → f32_  |  __2.15__ |
| `numkong.euclidean`    | _f64 → f32_  |  __2.14__ |
| ___SqEuclidean___      |              |           |
| `numkong.sqeuclidean`  | _u8 → f32_   |  __9.34__ |
| `numkong.sqeuclidean`  | _i8 → f32_   |  __8.60__ |
| `numkong.sqeuclidean`  | _bf16 → f32_ |  __0.93__ |
| `numkong.sqeuclidean`  | _f32 → f32_  |  __2.02__ |
| `numkong.sqeuclidean`  | _f64 → f32_  |  __2.14__ |

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
