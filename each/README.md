# Each Benchmarks

Elementwise sum and scale bandwidth benchmarks comparing NumKong against scalar baselines, ndarray, and nalgebra.

## Rust

| Library              | Precision     |       GB/s |
| :------------------- | :------------ | ---------: |
| ___Sum___            |               |            |
| `numkong::EachSum`   | _f32 → f32_   |  __90.85__ |
| `nalgebra::add`      | _f32 → f32_   |      88.76 |
| `ndarray::add`       | _f32 → f32_   |      88.33 |
| serial code          | _f32 → f32_   |      87.60 |
| serial code          | _f64 → f64_   |  __79.61__ |
| `ndarray::add`       | _f64 → f64_   |      79.08 |
| `nalgebra::add`      | _f64 → f64_   |      78.74 |
| `numkong::EachSum`   | _f64 → f64_   |      77.09 |
| `numkong::EachSum`   | _f16 → f16_   |  __89.93__ |
| `numkong::EachSum`   | _bf16 → bf16_ |  __16.51__ |
| `numkong::EachSum`   | _i8 → i8_     | __103.81__ |
| serial code          | _i8 → i8_     |     103.20 |
| ___Scale___          |               |            |
| serial code          | _f32 → f32_   |  __76.57__ |
| `ndarray::scale`     | _f32 → f32_   |      76.14 |
| `numkong::EachScale` | _f32 → f32_   |      61.99 |
| `nalgebra::scale`    | _f32 → f32_   |      36.81 |
| serial code          | _f64 → f64_   |  __67.48__ |
| `ndarray::scale`     | _f64 → f64_   |      67.42 |
| `numkong::EachScale` | _f64 → f64_   |      62.12 |
| `nalgebra::scale`    | _f64 → f64_   |      35.93 |
| `numkong::EachScale` | _f16 → f16_   |  __61.68__ |
| `numkong::EachScale` | _bf16 → bf16_ |  __30.91__ |
| serial code          | _i8 → i8_     |  __83.08__ |
| `numkong::EachScale` | _i8 → i8_     |      24.61 |

## Python

| Library       | Precision     |       GB/s |
| :------------ | :------------ | ---------: |
| ___Sum___     |               |            |
| `numpy.add`   | _i8 → i8_     |     133.70 |
| `numkong.add` | _i8 → i8_     | __115.27__ |
| `numkong.add` | _f32 → f32_   | __110.26__ |
| `numpy.add`   | _f32 → f32_   |     107.40 |
| `numpy.add`   | _f64 → f64_   |     106.52 |
| `numkong.add` | _f16 → f16_   | __99.922__ |
| `numkong.add` | _f64 → f64_   | __93.142__ |
| `numkong.add` | _bf16 → bf16_ |  __68.24__ |
| `numpy.add`   | _f16 → f16_   |       3.80 |

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
