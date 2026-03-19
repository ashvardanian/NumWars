# Similarities Benchmarks

All-pairs distance matrix benchmarks comparing NumKong packed kernels against ndarray and nalgebra.

## Rust

| Library                      | Precision    |      GSO/s |
| :--------------------------- | :----------- | ---------: |
| `numkong::angulars_packed`   | _u8 → f32_   | __694.88__ |
| `numkong::angulars_packed`   | _i8 → f32_   | __686.93__ |
| `numkong::euclideans_packed` | _i8 → f32_   | __685.67__ |
| `numkong::euclideans_packed` | _u8 → f32_   | __672.37__ |
| `numkong::angulars_packed`   | _bf16 → f32_ | __304.59__ |
| `numkong::euclideans_packed` | _bf16 → f32_ | __302.61__ |
| `ndarray angular`            | _f32 → f32_  |      38.20 |
| `nalgebra euclidean`         | _f32 → f32_  |      37.91 |
| `ndarray euclidean`          | _f32 → f32_  |      37.59 |
| `nalgebra angular`           | _f32 → f32_  |      36.97 |
| `numkong::euclideans_packed` | _f32 → f64_  |  __21.22__ |
| `numkong::angulars_packed`   | _f32 → f64_  |  __20.64__ |

## Python

| Library                     | Precision    |      GSO/s |
| :-------------------------- | :----------- | ---------: |
| `numkong.angulars_packed`   | _u8 → f32_   | __465.04__ |
| `numkong.euclideans_packed` | _u8 → f32_   | __463.47__ |
| `numkong.euclideans_packed` | _i8 → f32_   | __463.37__ |
| `numkong.angulars_packed`   | _i8 → f32_   | __454.74__ |
| `numkong.angulars_packed`   | _bf16 → f32_ | __226.56__ |
| `numkong.euclideans_packed` | _bf16 → f32_ | __210.12__ |
| `numkong.euclideans_packed` | _f32 → f64_  |  __20.24__ |
| `numkong.angulars_packed`   | _f32 → f64_  |  __19.84__ |
| `scipy.cdist`               | _f32 → f64_  |       2.83 |
| `scipy.cdist`               | _f32 → f64_  |       2.62 |

## Run It

### Rust

```bash
# Default 2048×2048 pairs at 2048 dimensions
cargo bench --bench bench_similarities --features bench_similarities

# Smaller 256×256 pairs at 256 dimensions
NUMWARS_DIMS=256 \
cargo bench --bench bench_similarities --features bench_similarities

# Focus on one metric
NUMWARS_FILTER="similarities/angulars/f32" \
cargo bench --bench bench_similarities --features bench_similarities
```

### Python

```bash
# Run the Python suite
python similarities/bench.py
```
