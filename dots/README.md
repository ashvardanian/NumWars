# Dots Benchmarks

Packed GEMM-style matrix multiplication benchmarks comparing NumKong against faer, matrixmultiply, ndarray, nalgebra, and NumPy.

## Rust

| Library          | Precision    |       GSO/s |
| :--------------- | :----------- | ----------: |
| `numkong`        | _i8 → i32_   | __1357.36__ |
| `numkong`        | _bf16 → f32_ |  __684.96__ |
| `numkong`        | _f16 → f32_  |  __106.63__ |
| `faer`           | _f32 → f32_  |   __81.21__ |
| `matrixmultiply` | _f32 → f32_  |       78.61 |
| `ndarray`        | _f32 → f32_  |       78.55 |
| `nalgebra`       | _f32 → f32_  |       74.21 |
| `numkong`        | _f32 → f64_  |   __42.04__ |

## Python

| Library   | Precision    |       GSO/s |
| :-------- | :----------- | ----------: |
| `numkong` | _i8 → i32_   | __1110.31__ |
| `numkong` | _bf16 → f32_ |  __487.89__ |
| `numpy`   | _f32 → f32_  |  __145.73__ |
| `numkong` | _f16 → f32_  |   __91.80__ |
| `numkong` | _f32 → f64_  |   __42.69__ |

## Run It

### Rust

```bash
# Default 2048×2048×2048 workload
cargo bench --bench bench_dots --features bench_dots

# Smaller 512×512×512 workload
NUMWARS_DIMS_WIDTH=512 NUMWARS_DIMS_HEIGHT=512 NUMWARS_DIMS_DEPTH=512 \
cargo bench --bench bench_dots --features bench_dots

# Focus on float32
NUMWARS_FILTER="dots/f32" \
cargo bench --bench bench_dots --features bench_dots
```

### Python

```bash
# Default 2048×2048×2048 workload, float32 only
python dots/bench.py --filter 'dots/numpy/f32/2048x2048x2048'
```
