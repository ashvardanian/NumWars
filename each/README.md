# Each Benchmarks

Elementwise sum, scale and fused multiply-add bandwidth benchmarks comparing NumKong against scalar baselines, ndarray, nalgebra, and NumPy.
Scale computes α × a + β, and FMA computes α × a × b + β × c.
The unit is GB/s.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
Every output keeps its input type, so NumKong widens nothing here.
The `baseline_*` alternatives are the hand-written serial loops of `bench.rs`.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `nalgebra::add`, `ndarray::add`, `nalgebra::scale`, `ndarray::fma`, `nalgebra::fma`.

| NumKong                                         |     Speed | Speedup | Alternative      |     Speed |
| :---------------------------------------------- | --------: | ------: | :--------------- | --------: |
| `numkong::EachSum::each_sum`, _f32 → f32_       |  251 GB/s |   1.34× | `baseline_sum`   |  188 GB/s |
| `numkong::EachSum::each_sum`, _f64 → f64_       |  148 GB/s |   0.97× | `baseline_sum`   |  153 GB/s |
| `numkong::EachSum::each_sum`, _f16 → f16_       |  154 GB/s |       – | –                |         – |
| `numkong::EachSum::each_sum`, _bf16 → bf16_     | 29.4 GB/s |       – | –                |         – |
| `numkong::EachSum::each_sum`, _i8 → i8_         |  179 GB/s |   4.02× | `baseline_sum`   | 44.4 GB/s |
| `numkong::EachScale::each_scale`, _f32 → f32_   | 98.1 GB/s |   1.03× | `ndarray::scale` | 95.6 GB/s |
| `numkong::EachScale::each_scale`, _f64 → f64_   | 91.1 GB/s |   0.95× | `ndarray::scale` | 95.8 GB/s |
| `numkong::EachScale::each_scale`, _f16 → f16_   | 49.6 GB/s |       – | –                |         – |
| `numkong::EachScale::each_scale`, _bf16 → bf16_ | 17.0 GB/s |       – | –                |         – |
| `numkong::EachScale::each_scale`, _i8 → i8_     | 10.2 GB/s |   0.25× | `baseline_scale` | 41.1 GB/s |
| `numkong::EachFma::each_fma`, _f32 → f32_       |  210 GB/s |   1.49× | `baseline_fma`   |  141 GB/s |
| `numkong::EachFma::each_fma`, _f64 → f64_       | 80.9 GB/s |   0.91× | `baseline_fma`   | 89.1 GB/s |
| `numkong::EachFma::each_fma`, _f16 → f16_       | 59.6 GB/s |       – | –                |         – |
| `numkong::EachFma::each_fma`, _bf16 → bf16_     | 33.2 GB/s |       – | –                |         – |
| `numkong::EachFma::each_fma`, _i8 → i8_         | 19.3 GB/s |   0.50× | `baseline_fma`   | 38.4 GB/s |

> Measured October 14, 2026.

### Python

| NumKong                        |     Speed | Speedup | Alternative                  |     Speed |
| :----------------------------- | --------: | ------: | :--------------------------- | --------: |
| `numkong.add`, _i8 → i8_       | 10.3 GB/s |   0.69× | `numpy.add`                  | 14.9 GB/s |
| `numkong.add`, _f32 → f32_     | 34.8 GB/s |   0.76× | `numpy.add`                  | 45.9 GB/s |
| `numkong.add`, _f16 → f16_     | 19.9 GB/s |   19.7× | `numpy.add`                  | 1.01 GB/s |
| `numkong.add`, _f64 → f64_     | 45.7 GB/s |   1.26× | `numpy.add`                  | 36.4 GB/s |
| `numkong.add`, _bf16 → bf16_   | 2.54 GB/s |       – | –                            |         – |
| `numkong.scale`, _f32 → f32_   | 31.9 GB/s |   3.47× | `numpy.multiply → numpy.add` | 9.20 GB/s |
| `numkong.scale`, _f64 → f64_   | 39.8 GB/s |   2.45× | `numpy.multiply → numpy.add` | 16.2 GB/s |
| `numkong.scale`, _f16 → f16_   | 15.5 GB/s |   48.3× | `numpy.multiply → numpy.add` | 0.32 GB/s |
| `numkong.scale`, _bf16 → bf16_ | 3.03 GB/s |       – | –                            |         – |
| `numkong.scale`, _i8 → i8_     | 6.64 GB/s |   2.66× | `numpy.multiply → numpy.add` | 2.50 GB/s |
| `numkong.fma`, _f32 → f32_     | 41.0 GB/s |   3.55× | `numpy.multiply → numpy.add` | 11.5 GB/s |
| `numkong.fma`, _f64 → f64_     | 55.1 GB/s |   3.17× | `numpy.multiply → numpy.add` | 17.4 GB/s |
| `numkong.fma`, _f16 → f16_     | 20.2 GB/s |   59.5× | `numpy.multiply → numpy.add` | 0.34 GB/s |
| `numkong.fma`, _bf16 → bf16_   | 2.42 GB/s |       – | –                            |         – |
| `numkong.fma`, _i8 → i8_       | 8.95 GB/s |   2.59× | `numpy.multiply → numpy.add` | 3.45 GB/s |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `ndarray::add`, `ndarray::scale`, `nalgebra::scale`.

| NumKong                                         |     Speed | Speedup | Alternative      |     Speed |
| :---------------------------------------------- | --------: | ------: | :--------------- | --------: |
| `numkong::EachSum::each_sum`, _f32 → f32_       | 90.9 GB/s |   1.02× | `nalgebra::add`  | 88.8 GB/s |
| `numkong::EachSum::each_sum`, _f64 → f64_       | 77.1 GB/s |   0.97× | `baseline_sum`   | 79.6 GB/s |
| `numkong::EachSum::each_sum`, _f16 → f16_       | 89.9 GB/s |       – | –                |         – |
| `numkong::EachSum::each_sum`, _bf16 → bf16_     | 16.5 GB/s |       – | –                |         – |
| `numkong::EachSum::each_sum`, _i8 → i8_         |  104 GB/s |   1.01× | `baseline_sum`   |  103 GB/s |
| `numkong::EachScale::each_scale`, _f32 → f32_   | 62.0 GB/s |   0.81× | `baseline_scale` | 76.6 GB/s |
| `numkong::EachScale::each_scale`, _f64 → f64_   | 62.1 GB/s |   0.92× | `baseline_scale` | 67.5 GB/s |
| `numkong::EachScale::each_scale`, _f16 → f16_   | 61.7 GB/s |       – | –                |         – |
| `numkong::EachScale::each_scale`, _bf16 → bf16_ | 30.9 GB/s |       – | –                |         – |
| `numkong::EachScale::each_scale`, _i8 → i8_     | 24.6 GB/s |   0.30× | `baseline_scale` | 83.1 GB/s |

> Measured October 1, 2026.

### Python

| NumKong                      |     Speed | Speedup | Alternative |     Speed |
| :--------------------------- | --------: | ------: | :---------- | --------: |
| `numkong.add`, _i8 → i8_     |  115 GB/s |   0.86× | `numpy.add` |  134 GB/s |
| `numkong.add`, _f32 → f32_   |  110 GB/s |   1.03× | `numpy.add` |  107 GB/s |
| `numkong.add`, _f16 → f16_   | 99.9 GB/s |   26.3× | `numpy.add` | 3.80 GB/s |
| `numkong.add`, _f64 → f64_   | 93.1 GB/s |   0.87× | `numpy.add` |  107 GB/s |
| `numkong.add`, _bf16 → bf16_ | 68.2 GB/s |       – | –           |         – |

> Measured October 1, 2026.

## Run It

### Rust

```bash
# Default 2048-element tensors
cargo bench --bench bench_each --features bench_each

# Focus on one operation family
NUMWARS_FILTER="each/sum|each/scale" \
cargo bench --bench bench_each --features bench_each
```

### Python

```bash
# Default 2048-element tensors, add on f32
NUMWARS_FILTER='each/add/f32' python each/bench.py
```
