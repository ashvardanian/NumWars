# Reduce Benchmarks

Horizontal reduction and summary-statistics benchmarks comparing NumKong against Polars, ndarray, and scalar baselines.

## Rust

| Library                 |      Precision      |      GB/s |
| :---------------------- | :-----------------: | --------: |
| `polars sum`            | _f32 → Option<f32>_ | __43.86__ |
| `numkong moments().sum` |     _f32 → f64_     |     43.26 |
| `ndarray sum`           |     _f32 → f32_     |     36.38 |
| `scalar sum`            |     _f32 → f32_     |      6.67 |

## Python

| Library      | Operation |  Precision  |      GB/s |
| :----------- | :-------: | :---------: | --------: |
| `numpy sum`  |    sum    | _f64 → f64_ | __24.75__ |
| `numpy sum`  |    sum    | _f32 → f32_ |     18.01 |
| `numpy norm` |   norm    | _f64 → f64_ |      7.52 |
| `numpy norm` |   norm    | _f32 → f64_ |      6.49 |
| `numpy sum`  |    sum    |  _i8 → i8_  |      2.81 |

## Run It

### Rust

```bash
# Default 1M-element tensors
cargo bench --bench bench_reduce --features bench_reduce

# Smaller 10K-element tensors
NUMWARS_DIMS=10000 \
cargo bench --bench bench_reduce --features bench_reduce

# Focus on one operation
NUMWARS_FILTER="reduce/minmax" \
cargo bench --bench bench_reduce --features bench_reduce
```

### Python

```bash
python reduce/bench.py
```
