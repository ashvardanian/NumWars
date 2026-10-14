# Similarities Benchmarks

All-pairs distance matrix benchmarks comparing NumKong packed kernels against ndarray and nalgebra.
The SYRK rows score one matrix against itself, A × Aᵀ, where NumKong's `*_symmetric` kernels fill the upper triangle.
In Rust, ndarray and nalgebra compute the full Gram matrix and apply the distance formula to its upper triangle, reading the norms off its diagonal, and a serial `count_ones` loop over 64-bit words is the bit-set baseline.
In Python, SciPy's `pdist` returns the condensed upper triangle without the diagonal, and its Hamming distance is the fraction of differing bits rather than their count.
Every SYRK row counts only the n(n+1)/2 useful pairs of one triangle times the depth, so a full product earns no credit for the mirrored half.

## Rust

| Library                         | Precision    |      GSO/s |
| :------------------------------ | :----------- | ---------: |
| ___Angular___                   |              |            |
| `numkong::angulars_packed`      | _i8 → f32_   | __830.13__ |
| `numkong::angulars_packed`      | _u8 → f32_   | __830.14__ |
| `numkong::angulars_packed`      | _bf16 → f32_ | __502.45__ |
| `numkong::angulars_packed`      | _f32 → f64_  |  __92.52__ |
| `ndarray angular`               | _f32 → f32_  |      56.98 |
| `nalgebra angular`              | _f32 → f32_  |      49.95 |
| `ndarray angular`               | _f64 → f64_  |      28.82 |
| `nalgebra angular`              | _f64 → f64_  |      27.26 |
| `numkong::angulars_packed`      | _f64 → f64_  |  __22.81__ |
| `numkong::angulars_symmetric`   | _i8 → f32_   |            |
| `numkong::angulars_symmetric`   | _u8 → f32_   |            |
| `numkong::angulars_symmetric`   | _bf16 → f32_ |            |
| `numkong::angulars_symmetric`   | _f32 → f64_  |            |
| `ndarray A × Aᵀ angular`        | _f32 → f32_  |            |
| `nalgebra A × Aᵀ angular`       | _f32 → f32_  |            |
| `ndarray A × Aᵀ angular`        | _f64 → f64_  |            |
| `nalgebra A × Aᵀ angular`       | _f64 → f64_  |            |
| `numkong::angulars_symmetric`   | _f64 → f64_  |            |
| ___Euclidean___                 |              |            |
| `numkong::euclideans_packed`    | _i8 → f32_   | __887.85__ |
| `numkong::euclideans_packed`    | _u8 → f32_   | __888.74__ |
| `numkong::euclideans_packed`    | _bf16 → f32_ | __524.00__ |
| `numkong::euclideans_packed`    | _f32 → f64_  |  __92.93__ |
| `ndarray euclidean`             | _f32 → f32_  |      57.64 |
| `nalgebra euclidean`            | _f32 → f32_  |      49.79 |
| `ndarray euclidean`             | _f64 → f64_  |      28.82 |
| `nalgebra euclidean`            | _f64 → f64_  |      27.11 |
| `numkong::euclideans_packed`    | _f64 → f64_  |  __22.85__ |
| `numkong::euclideans_symmetric` | _i8 → f32_   |            |
| `numkong::euclideans_symmetric` | _u8 → f32_   |            |
| `numkong::euclideans_symmetric` | _bf16 → f32_ |            |
| `numkong::euclideans_symmetric` | _f32 → f64_  |            |
| `ndarray A × Aᵀ euclidean`      | _f32 → f32_  |            |
| `nalgebra A × Aᵀ euclidean`     | _f32 → f32_  |            |
| `ndarray A × Aᵀ euclidean`      | _f64 → f64_  |            |
| `nalgebra A × Aᵀ euclidean`     | _f64 → f64_  |            |
| `numkong::euclideans_symmetric` | _f64 → f64_  |            |
| ___Hamming___                   |              |            |
| `numkong::hammings_packed`      | _u1x8_       |   __9821__ |
| `numkong::hammings_symmetric`   | _u1x8_       |            |
| `serial count_ones`             | _u1x8_       |            |
| ___Jaccard___                   |              |            |
| `numkong::jaccards_packed`      | _u1x8_       |   __3173__ |
| `numkong::jaccards_symmetric`   | _u1x8_       |            |
| `serial count_ones`             | _u1x8_       |            |

## Python

| Library                        | Precision    |      GSO/s |
| :----------------------------- | :----------- | ---------: |
| `numkong.euclideans_packed`    | _u8 → f32_   | __425.91__ |
| `numkong.euclideans_packed`    | _i8 → f32_   | __408.64__ |
| `numkong.angulars_packed`      | _i8 → f32_   | __386.96__ |
| `numkong.angulars_packed`      | _u8 → f32_   | __364.01__ |
| `numkong.angulars_packed`      | _f32 → f64_  |  __79.26__ |
| `numkong.euclideans_packed`    | _f32 → f64_  |  __52.95__ |
| `scipy.cdist euclidean`        | _f32 → f64_  |       5.09 |
| `scipy.cdist cosine`           | _f32 → f64_  |       1.30 |
| `numkong.euclideans_symmetric` | _u8 → f32_   |            |
| `numkong.euclideans_symmetric` | _i8 → f32_   |            |
| `numkong.angulars_symmetric`   | _i8 → f32_   |            |
| `numkong.angulars_symmetric`   | _u8 → f32_   |            |
| `numkong.angulars_symmetric`   | _f32 → f64_  |            |
| `numkong.euclideans_symmetric` | _f32 → f64_  |            |
| `numkong.hammings_symmetric`   | _u1 → u32_   |            |
| `numkong.jaccards_symmetric`   | _u1 → f32_   |            |
| `scipy.pdist euclidean`        | _f32 → f64_  |            |
| `scipy.pdist cosine`           | _f32 → f64_  |            |
| `scipy.pdist hamming`          | _bool → f64_ |            |
| `scipy.pdist jaccard`          | _bool → f64_ |            |

## Run It

### Rust

```bash
# Default 2048×2048 pairs at 1536 dimensions
cargo bench --bench bench_similarities --features bench_similarities

# Smaller 256×256 pairs at 256 dimensions
NUMWARS_DIMS=256 NUMWARS_BATCH_PER_CORE=256 \
cargo bench --bench bench_similarities --features bench_similarities

# Focus on one metric
NUMWARS_FILTER="similarities/angulars/f32" \
cargo bench --bench bench_similarities --features bench_similarities

# Only the SYRK rows, 2048×2048 matrices with one triangle each
NUMWARS_FILTER="similarities/symmetric" \
cargo bench --bench bench_similarities --features bench_similarities
```

### Python

```bash
# Run the Python suite
python similarities/bench.py

# Only the SYRK rows
NUMWARS_FILTER='similarities/symmetric' python similarities/bench.py
```
