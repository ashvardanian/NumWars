# Similarities Benchmarks

All-pairs distance matrix benchmarks comparing NumKong packed kernels against ndarray and nalgebra.

## Rust

| Library                      | Precision    |      GSO/s |
| :--------------------------- | :----------- | ---------: |
| ___Angular___                |              |            |
| `numkong::angulars_packed`   | _i8 → f32_   | __830.13__ |
| `numkong::angulars_packed`   | _u8 → f32_   | __830.14__ |
| `numkong::angulars_packed`   | _bf16 → f32_ | __502.45__ |
| `numkong::angulars_packed`   | _f32 → f64_  |  __92.52__ |
| `ndarray angular`            | _f32 → f32_  |      56.98 |
| `nalgebra angular`           | _f32 → f32_  |      49.95 |
| `ndarray angular`            | _f64 → f64_  |      28.82 |
| `nalgebra angular`           | _f64 → f64_  |      27.26 |
| `numkong::angulars_packed`   | _f64 → f64_  |  __22.81__ |
| ___Euclidean___              |              |            |
| `numkong::euclideans_packed` | _i8 → f32_   | __887.85__ |
| `numkong::euclideans_packed` | _u8 → f32_   | __888.74__ |
| `numkong::euclideans_packed` | _bf16 → f32_ | __524.00__ |
| `numkong::euclideans_packed` | _f32 → f64_  |  __92.93__ |
| `ndarray euclidean`          | _f32 → f32_  |      57.64 |
| `nalgebra euclidean`         | _f32 → f32_  |      49.79 |
| `ndarray euclidean`          | _f64 → f64_  |      28.82 |
| `nalgebra euclidean`         | _f64 → f64_  |      27.11 |
| `numkong::euclideans_packed` | _f64 → f64_  |  __22.85__ |
| ___Hamming___                |              |            |
| `numkong::hammings_packed`   | _u1x8_       |   __9821__ |
| ___Jaccard___                |              |            |
| `numkong::jaccards_packed`   | _u1x8_       |   __3173__ |

## Python

| Library                     | Precision   |      GSO/s |
| :-------------------------- | :---------- | ---------: |
| `numkong.euclideans_packed` | _u8 → f32_  | __425.91__ |
| `numkong.euclideans_packed` | _i8 → f32_  | __408.64__ |
| `numkong.angulars_packed`   | _i8 → f32_  | __386.96__ |
| `numkong.angulars_packed`   | _u8 → f32_  | __364.01__ |
| `numkong.angulars_packed`   | _f32 → f64_ |  __79.26__ |
| `numkong.euclideans_packed` | _f32 → f64_ |  __52.95__ |
| `scipy.cdist euclidean`     | _f32 → f64_ |       5.09 |
| `scipy.cdist cosine`        | _f32 → f64_ |       1.30 |

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
