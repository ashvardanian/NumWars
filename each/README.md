# Each Benchmarks

Elementwise sum and scale bandwidth benchmarks comparing NumKong against scalar baselines, ndarray, and nalgebra.

## Rust

| Library              | Precision     |      GB/s |
| :------------------- | :------------ | --------: |
| ___Sum___            |               |           |
| serial code          | _f32 → f32_   | __19.86__ |
| `nalgebra::add`      | _f32 → f32_   |     19.82 |
| `ndarray::add`       | _f32 → f32_   |     19.79 |
| `numkong::EachSum`   | _f32 → f32_   |     18.73 |
| `nalgebra::add`      | _f64 → f64_   | __20.24__ |
| serial code          | _f64 → f64_   |     20.10 |
| `ndarray::add`       | _f64 → f64_   |     19.89 |
| `numkong::EachSum`   | _f64 → f64_   |     19.53 |
| `numkong::EachSum`   | _f16 → f16_   | __19.93__ |
| `numkong::EachSum`   | _bf16 → bf16_ | __19.08__ |
| serial code          | _i8 → i8_     | __23.40__ |
| `numkong::EachSum`   | _i8 → i8_     |     23.23 |
| ___Scale___          |               |           |
| serial code          | _f32 → f32_   | __14.64__ |
| `ndarray::scale`     | _f32 → f32_   |     14.49 |
| `numkong::EachScale` | _f32 → f32_   |     13.90 |
| `nalgebra::scale`    | _f32 → f32_   |      9.08 |
| serial code          | _f64 → f64_   | __14.39__ |
| `ndarray::scale`     | _f64 → f64_   |     14.14 |
| `numkong::EachScale` | _f64 → f64_   |     13.88 |
| `nalgebra::scale`    | _f64 → f64_   |      9.08 |
| `numkong::EachScale` | _bf16 → bf16_ | __14.28__ |
| `numkong::EachScale` | _f16 → f16_   | __13.13__ |
| serial code          | _i8 → i8_     | __23.58__ |
| `numkong::EachScale` | _i8 → i8_     |     22.64 |

## Python

| Library       | Precision     |      GB/s |
| :------------ | :------------ | --------: |
| ___Sum___     |               |           |
| `numpy.add`   | _i8 → i8_     | __33.32__ |
| `numkong.add` | _i8 → i8_     |     30.91 |
| `numkong.add` | _f32 → f32_   | __29.39__ |
| `numkong.add` | _f16 → f16_   | __28.84__ |
| `numkong.add` | _f64 → f64_   | __28.79__ |
| `numkong.add` | _bf16 → bf16_ | __27.72__ |
| `numpy.add`   | _f32 → f32_   |     25.65 |
| `numpy.add`   | _f64 → f64_   |     25.03 |
| `numpy.add`   | _f16 → f16_   |      0.95 |

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
