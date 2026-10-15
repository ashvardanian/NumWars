# Similarity Benchmarks

Pairwise vector-vector distance and dot-product benchmarks comparing NumKong against scalar baselines and BLAS-backed libraries.
GSO/s counts one scalar operation per dimension of the pair, as the Python harness does, and the Rust rows derive it from Criterion's time per call.

Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong widens its outputs: i8 → i32, bf16, f16 and the FP8 and FP6 formats → f32, f32 → f64.
The `baseline_*` alternatives are the hand-written serial loops of `bench.rs`.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `nalgebra::dot`, `ndarray::norm`, `nalgebra::norm_squared`, `ndarray_stats::kl_divergence`.

| NumKong                                                  |      Speed | Speedup | Alternative                             |      Speed |
| :------------------------------------------------------- | ---------: | ------: | :-------------------------------------- | ---------: |
| `numkong::Dot::dot`, _f32 → f64_                         | 7.80 GSO/s |   0.23× | `cblas_sdot`, _f32 → f32_               | 34.4 GSO/s |
| `numkong::Dot::dot`, _f64 → f64_                         | 5.15 GSO/s |   0.30× | `cblas_ddot`                            | 17.4 GSO/s |
| `numkong::Dot::dot`, _i8 → i32_                          | 58.6 GSO/s |   11.5× | `baseline_dot`                          | 5.09 GSO/s |
| `numkong::Dot::dot`, _u8 → u32_                          | 65.9 GSO/s |   13.0× | `baseline_dot`                          | 5.07 GSO/s |
| `numkong::Dot::dot`, _f16 → f32_                         | 15.7 GSO/s |   98.4× | `baseline_dot`                          | 0.16 GSO/s |
| `numkong::Dot::dot`, _bf16 → f32_                        | 23.1 GSO/s |    105× | `baseline_dot`                          | 0.22 GSO/s |
| `numkong::Dot::dot`, _e4m3 → f32_                        | 9.41 GSO/s |   47.1× | `baseline_dot`                          | 0.20 GSO/s |
| `numkong::Dot::dot`, _e5m2 → f32_                        | 6.78 GSO/s |   30.8× | `baseline_dot`                          | 0.22 GSO/s |
| `numkong::Dot::dot`, _e2m3 → f32_                        | 44.2 GSO/s |    201× | `baseline_dot`                          | 0.22 GSO/s |
| `numkong::Dot::dot`, _e3m2 → f32_                        | 20.1 GSO/s |    101× | `baseline_dot`                          | 0.20 GSO/s |
| `numkong::Angular::angular`, _f32 → f64_                 | 6.80 GSO/s |   2.29× | `baseline_angular`, _f32 → f32_         | 2.97 GSO/s |
| `numkong::Angular::angular`, _f64 → f64_                 | 1.66 GSO/s |   0.72× | `baseline_angular`                      | 2.32 GSO/s |
| `numkong::Angular::angular`, _i8 → f32_                  | 23.8 GSO/s |   7.27× | `baseline_angular`                      | 3.27 GSO/s |
| `numkong::Angular::angular`, _u8 → f32_                  | 32.0 GSO/s |   7.57× | `baseline_angular`                      | 4.23 GSO/s |
| `numkong::Angular::angular`, _f16 → f32_                 | 12.7 GSO/s |   84.5× | `baseline_angular`                      | 0.15 GSO/s |
| `numkong::Angular::angular`, _bf16 → f32_                | 14.5 GSO/s |   68.9× | `baseline_angular`                      | 0.21 GSO/s |
| `numkong::Angular::angular`, _e4m3 → f32_                | 7.29 GSO/s |   38.4× | `baseline_angular`                      | 0.19 GSO/s |
| `numkong::Angular::angular`, _e5m2 → f32_                | 11.6 GSO/s |   55.0× | `baseline_angular`                      | 0.21 GSO/s |
| `numkong::Angular::angular`, _e2m3 → f32_                | 20.7 GSO/s |    109× | `baseline_angular`                      | 0.19 GSO/s |
| `numkong::Angular::angular`, _e3m2 → f32_                | 11.7 GSO/s |   61.3× | `baseline_angular`                      | 0.19 GSO/s |
| `numkong::Euclidean::euclidean`, _f32 → f64_             | 6.12 GSO/s |   0.63× | `nalgebra::norm`, _f32 → f32_           | 9.72 GSO/s |
| `numkong::Euclidean::euclidean`, _f64 → f64_             | 3.95 GSO/s |   0.77× | `nalgebra::norm`                        | 5.12 GSO/s |
| `numkong::Euclidean::euclidean`, _i8 → f32_              | 40.1 GSO/s |   8.24× | `baseline_euclidean`                    | 4.87 GSO/s |
| `numkong::Euclidean::euclidean`, _u8 → f32_              | 44.7 GSO/s |   9.12× | `baseline_euclidean`                    | 4.90 GSO/s |
| `numkong::Euclidean::euclidean`, _f16 → f32_             | 15.4 GSO/s |   96.2× | `baseline_euclidean`                    | 0.16 GSO/s |
| `numkong::Euclidean::euclidean`, _bf16 → f32_            | 15.0 GSO/s |   68.1× | `baseline_euclidean`                    | 0.22 GSO/s |
| `numkong::Euclidean::euclidean`, _e4m3 → f32_            | 7.53 GSO/s |   37.7× | `baseline_euclidean`                    | 0.20 GSO/s |
| `numkong::Euclidean::euclidean`, _e5m2 → f32_            | 13.6 GSO/s |   61.7× | `baseline_euclidean`                    | 0.22 GSO/s |
| `numkong::Euclidean::euclidean`, _e2m3 → f32_            | 36.3 GSO/s |    165× | `baseline_euclidean`                    | 0.22 GSO/s |
| `numkong::Euclidean::euclidean`, _e3m2 → f32_            | 18.5 GSO/s |   92.6× | `baseline_euclidean`                    | 0.20 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _f32 → f64_           | 7.31 GSO/s |   0.85× | `ndarray::dot`, _f32 → f32_             | 8.62 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _f64 → f64_           | 4.19 GSO/s |   0.81× | `ndarray::dot`                          | 5.20 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _i8 → u32_            | 46.0 GSO/s |   9.38× | `baseline_sqeuclidean`                  | 4.90 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _u8 → u32_            | 49.2 GSO/s |   10.0× | `baseline_sqeuclidean`                  | 4.90 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _f16 → f32_           | 15.6 GSO/s |   97.7× | `baseline_sqeuclidean`                  | 0.16 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _bf16 → f32_          | 15.1 GSO/s |   68.4× | `baseline_sqeuclidean`                  | 0.22 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _e4m3 → f32_          | 7.53 GSO/s |   37.7× | `baseline_sqeuclidean`                  | 0.20 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _e5m2 → f32_          | 13.7 GSO/s |   62.4× | `baseline_sqeuclidean`                  | 0.22 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _e2m3 → f32_          | 38.5 GSO/s |    175× | `baseline_sqeuclidean`                  | 0.22 GSO/s |
| `numkong::Euclidean::sqeuclidean`, _e3m2 → f32_          | 18.6 GSO/s |   92.8× | `baseline_sqeuclidean`                  | 0.20 GSO/s |
| `numkong::Hamming::hamming`, _u1x8 → u32_                |  258 GSO/s |   3.12× | `baseline_hamming_u1x8`                 | 82.6 GSO/s |
| `numkong::Jaccard::jaccard`, _u1x8 → f32_                |  196 GSO/s |   4.86× | `baseline_jaccard_u1x8`                 | 40.3 GSO/s |
| `numkong::KullbackLeibler::kullbackleibler`, _f32 → f64_ | 2.95 GSO/s |   5.90× | `baseline_kullbackleibler`, _f32 → f32_ | 0.50 GSO/s |
| `numkong::KullbackLeibler::kullbackleibler`, _f64 → f64_ | 0.92 GSO/s |   2.79× | `baseline_kullbackleibler`              | 0.33 GSO/s |
| `numkong::JensenShannon::jensenshannon`, _f32 → f64_     | 1.47 GSO/s |   5.25× | `baseline_jensenshannon`, _f32 → f32_   | 0.28 GSO/s |
| `numkong::JensenShannon::jensenshannon`, _f64 → f64_     | 0.44 GSO/s |   2.75× | `baseline_jensenshannon`                | 0.16 GSO/s |

> Measured October 14, 2026.

### Python

| NumKong                             |      Speed | Speedup | Alternative              |      Speed |
| :---------------------------------- | ---------: | ------: | :----------------------- | ---------: |
| `numkong.dot`, _u8 → f32_           | 5.88 GSO/s |       – | –                        |          – |
| `numkong.dot`, _i8 → f32_           | 5.48 GSO/s |       – | –                        |          – |
| `numkong.dot`, _bf16 → f32_         | 0.39 GSO/s |       – | –                        |          – |
| `numkong.dot`, _f32 → f32_          | 3.77 GSO/s |   2.14× | `numpy.dot`              | 1.76 GSO/s |
| `numkong.dot`, _f64 → f32_          | 2.80 GSO/s |   1.81× | `numpy.dot`, _f64 → f64_ | 1.55 GSO/s |
| `numkong.angular`, _u8 → f32_       | 5.86 GSO/s |       – | –                        |          – |
| `numkong.angular`, _i8 → f32_       | 5.21 GSO/s |       – | –                        |          – |
| `numkong.angular`, _bf16 → f32_     | 0.39 GSO/s |       – | –                        |          – |
| `numkong.angular`, _f32 → f32_      | 3.27 GSO/s |       – | –                        |          – |
| `numkong.angular`, _f64 → f32_      | 1.31 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _u8 → f32_     | 6.13 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _i8 → f32_     | 5.90 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _bf16 → f32_   | 0.40 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _f32 → f32_    | 3.63 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _f64 → f32_    | 2.51 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _u8 → f32_   | 3.00 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _i8 → f32_   | 3.09 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _bf16 → f32_ | 0.22 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _f32 → f32_  | 3.33 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _f64 → f32_  | 2.39 GSO/s |       – | –                        |          – |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `ndarray::norm`.

| NumKong                                                  |      Speed | Speedup | Alternative                             |      Speed |
| :------------------------------------------------------- | ---------: | ------: | :-------------------------------------- | ---------: |
| `numkong::Dot::dot`, _f32 → f64_                         | 2.50 GSO/s |   0.18× | `baseline_dot`, _f32 → f32_             | 13.9 GSO/s |
| `numkong::Dot::dot`, _f64 → f64_                         | 2.81 GSO/s |   0.24× | `nalgebra::dot`                         | 11.7 GSO/s |
| `numkong::Dot::dot`, _i8 → i32_                          | 37.4 GSO/s |   1.90× | `baseline_dot`                          | 19.7 GSO/s |
| `numkong::Dot::dot`, _u8 → u32_                          | 37.3 GSO/s |   1.94× | `baseline_dot`                          | 19.3 GSO/s |
| `numkong::Dot::dot`, _f16 → f32_                         | 6.16 GSO/s |       – | `baseline_dot`                          |          ⋯ |
| `numkong::Dot::dot`, _bf16 → f32_                        | 14.8 GSO/s |   22.1× | `baseline_dot`                          | 0.67 GSO/s |
| `numkong::Dot::dot`, _e4m3 → f32_                        | 4.43 GSO/s |       – | `baseline_dot`                          |          ⋯ |
| `numkong::Dot::dot`, _e5m2 → f32_                        | 6.08 GSO/s |       – | `baseline_dot`                          |          ⋯ |
| `numkong::Dot::dot`, _e2m3 → f32_                        | 26.6 GSO/s |       – | `baseline_dot`                          |          ⋯ |
| `numkong::Dot::dot`, _e3m2 → f32_                        | 11.3 GSO/s |       – | `baseline_dot`                          |          ⋯ |
| `numkong::Angular::angular`, _f32 → f64_                 | 2.34 GSO/s |       – | `baseline_angular`, _f32 → f32_         |          ⋯ |
| `numkong::Angular::angular`, _f64 → f64_                 | 2.30 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _i8 → f32_                  | 31.9 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _u8 → f32_                  | 31.9 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _f16 → f32_                 | 4.81 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _bf16 → f32_                | 10.3 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _e4m3 → f32_                | 2.19 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _e5m2 → f32_                | 5.34 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _e2m3 → f32_                | 2.60 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Angular::angular`, _e3m2 → f32_                | 2.60 GSO/s |       – | `baseline_angular`                      |          ⋯ |
| `numkong::Euclidean::euclidean`, _f32 → f64_             | 2.50 GSO/s |   0.22× | `baseline_euclidean`, _f32 → f32_       | 11.2 GSO/s |
| `numkong::Euclidean::euclidean`, _f64 → f64_             | 2.60 GSO/s |   0.46× | `nalgebra::norm`                        | 5.63 GSO/s |
| `numkong::Euclidean::euclidean`, _i8 → f32_              | 34.9 GSO/s |   3.77× | `baseline_euclidean`                    | 9.26 GSO/s |
| `numkong::Euclidean::euclidean`, _u8 → f32_              | 34.9 GSO/s |   3.79× | `baseline_euclidean`                    | 9.19 GSO/s |
| `numkong::Euclidean::euclidean`, _f16 → f32_             | 5.27 GSO/s |       – | `baseline_euclidean`                    |          ⋯ |
| `numkong::Euclidean::euclidean`, _bf16 → f32_            | 5.54 GSO/s |   8.39× | `baseline_euclidean`                    | 0.66 GSO/s |
| `numkong::Euclidean::euclidean`, _e4m3 → f32_            | 2.28 GSO/s |       – | `baseline_euclidean`                    |          ⋯ |
| `numkong::Euclidean::euclidean`, _e5m2 → f32_            | 5.80 GSO/s |       – | `baseline_euclidean`                    |          ⋯ |
| `numkong::Euclidean::euclidean`, _e2m3 → f32_            | 2.74 GSO/s |       – | `baseline_euclidean`                    |          ⋯ |
| `numkong::Euclidean::euclidean`, _e3m2 → f32_            | 2.75 GSO/s |       – | `baseline_euclidean`                    |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _f32 → f64_           | 2.65 GSO/s |       – | `ndarray::dot`, _f32 → f32_             |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _f64 → f64_           | 2.76 GSO/s |       – | `ndarray::dot`                          |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _i8 → u32_            | 38.6 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _u8 → u32_            | 38.8 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _f16 → f32_           | 5.52 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _bf16 → f32_          | 5.70 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _e4m3 → f32_          | 2.29 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _e5m2 → f32_          | 5.87 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _e2m3 → f32_          | 2.76 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Euclidean::sqeuclidean`, _e3m2 → f32_          | 2.76 GSO/s |       – | `baseline_sqeuclidean`                  |          ⋯ |
| `numkong::Hamming::hamming`, _u1x8 → u32_                | 45.2 GSO/s |       – | `baseline_hamming_u1x8`                 |          ⋯ |
| `numkong::Jaccard::jaccard`, _u1x8 → f32_                | 36.9 GSO/s |       – | `baseline_jaccard_u1x8`                 |          ⋯ |
| `numkong::KullbackLeibler::kullbackleibler`, _f32 → f64_ | 2.87 GSO/s |       – | `baseline_kullbackleibler`, _f32 → f32_ |          ⋯ |
| `numkong::KullbackLeibler::kullbackleibler`, _f64 → f64_ | 0.27 GSO/s |       – | `baseline_kullbackleibler`              |          ⋯ |
| `numkong::JensenShannon::jensenshannon`, _f32 → f64_     | 1.47 GSO/s |       – | `baseline_jensenshannon`, _f32 → f32_   |          ⋯ |
| `numkong::JensenShannon::jensenshannon`, _f64 → f64_     | 0.17 GSO/s |       – | `baseline_jensenshannon`                |          ⋯ |

> Measured March 22, 2026.

### Python

| NumKong                             |      Speed | Speedup | Alternative              |      Speed |
| :---------------------------------- | ---------: | ------: | :----------------------- | ---------: |
| `numkong.dot`, _u8 → f32_           | 9.37 GSO/s |       – | –                        |          – |
| `numkong.dot`, _i8 → f32_           | 9.81 GSO/s |       – | –                        |          – |
| `numkong.dot`, _bf16 → f32_         | 1.06 GSO/s |       – | –                        |          – |
| `numkong.dot`, _f32 → f32_          | 2.02 GSO/s |   0.53× | `numpy.dot`              | 3.81 GSO/s |
| `numkong.dot`, _f64 → f32_          | 2.48 GSO/s |   0.67× | `numpy.dot`, _f64 → f64_ | 3.68 GSO/s |
| `numkong.angular`, _u8 → f32_       | 10.1 GSO/s |       – | –                        |          – |
| `numkong.angular`, _i8 → f32_       | 10.0 GSO/s |       – | –                        |          – |
| `numkong.angular`, _bf16 → f32_     | 1.03 GSO/s |       – | –                        |          – |
| `numkong.angular`, _f32 → f32_      | 1.98 GSO/s |       – | –                        |          – |
| `numkong.angular`, _f64 → f32_      | 2.01 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _u8 → f32_     | 10.2 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _i8 → f32_     | 10.3 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _bf16 → f32_   | 0.92 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _f32 → f32_    | 2.15 GSO/s |       – | –                        |          – |
| `numkong.euclidean`, _f64 → f32_    | 2.14 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _u8 → f32_   | 9.34 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _i8 → f32_   | 8.60 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _bf16 → f32_ | 0.93 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _f32 → f32_  | 2.02 GSO/s |       – | –                        |          – |
| `numkong.sqeuclidean`, _f64 → f32_  | 2.14 GSO/s |       – | –                        |          – |

> Measured March 22, 2026.
## Run It

### Rust

```bash
# Default 1536-dimensional vectors
cargo bench --bench bench_similarity --features bench_similarity

# Smaller 512-dimensional vectors
NUMWARS_DIMS=512 cargo bench --bench bench_similarity --features bench_similarity

# One benchmark group
NUMWARS_FILTER="similarity/angular/f32" \
cargo bench --bench bench_similarity --features bench_similarity
```

### Python

```bash
# Default 1536-dimensional pairwise distances
NUMWARS_FILTER='angular/f32' python similarity/bench.py

# Shorter vectors
NUMWARS_DIMS=256 NUMWARS_FILTER='dot' python similarity/bench.py
```
