# Each Benchmarks

Elementwise sum and scale bandwidth benchmarks comparing NumKong against scalar baselines, ndarray, and nalgebra.

## Rust

| Library              | Precision     |       GB/s |
| :------------------- | :------------ | ---------: |
| ___Sum___            |               |            |
| `numkong::EachSum`   | _f32 → f32_   |  __97.55__ |
| `nalgebra::add`      | _f32 → f32_   |      95.31 |
| `ndarray::add`       | _f32 → f32_   |      94.84 |
| serial code          | _f32 → f32_   |      94.06 |
| serial code          | _f64 → f64_   |  __85.48__ |
| `ndarray::add`       | _f64 → f64_   |      84.91 |
| `nalgebra::add`      | _f64 → f64_   |      84.55 |
| `numkong::EachSum`   | _f64 → f64_   |      82.77 |
| `numkong::EachSum`   | _f16 → f16_   |  __96.56__ |
| `numkong::EachSum`   | _bf16 → bf16_ |  __17.73__ |
| `numkong::EachSum`   | _i8 → i8_     | __111.47__ |
| serial code          | _i8 → i8_     |     110.81 |
| ___Scale___          |               |            |
| serial code          | _f32 → f32_   |  __82.22__ |
| `ndarray::scale`     | _f32 → f32_   |      81.75 |
| `numkong::EachScale` | _f32 → f32_   |      66.56 |
| `nalgebra::scale`    | _f32 → f32_   |      39.52 |
| serial code          | _f64 → f64_   |  __72.46__ |
| `ndarray::scale`     | _f64 → f64_   |      72.39 |
| `numkong::EachScale` | _f64 → f64_   |      66.70 |
| `nalgebra::scale`    | _f64 → f64_   |      38.58 |
| `numkong::EachScale` | _f16 → f16_   |  __66.23__ |
| `numkong::EachScale` | _bf16 → bf16_ |  __33.19__ |
| serial code          | _i8 → i8_     |  __89.21__ |
| `numkong::EachScale` | _i8 → i8_     |      26.43 |

## Python

| Library       | Precision     |       GB/s |
| :------------ | :------------ | ---------: |
| ___Sum___     |               |            |
| `numpy.add`   | _i8 → i8_     |     143.56 |
| `numkong.add` | _i8 → i8_     | __123.77__ |
| `numkong.add` | _f32 → f32_   | __118.39__ |
| `numpy.add`   | _f32 → f32_   |     115.32 |
| `numpy.add`   | _f64 → f64_   |     114.37 |
| `numkong.add` | _f16 → f16_   | __107.29__ |
| `numkong.add` | _f64 → f64_   | __100.01__ |
| `numkong.add` | _bf16 → bf16_ |  __73.27__ |
| `numpy.add`   | _f16 → f16_   |       4.08 |

## Run It

### Rust

```bash
# Default 1M-element tensors
cargo bench --bench bench_each --features bench_each

# Focus on one operation family
NUMWARS_FILTER="each/sum|each/scale" \
cargo bench --bench bench_each --features bench_each
```

### Python

```bash
# Default 1M-element tensors, add on float32
python each/bench.py --filter 'add/float32'
```
