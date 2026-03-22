# Dots Benchmarks

Packed GEMM-style matrix multiplication benchmarks comparing NumKong against faer, matrixmultiply, ndarray, nalgebra, and NumPy.

## Rust

| Library                         | Precision    |       GSO/s |
| :------------------------------ | :----------- | ----------: |
| `numkong::try_dots_packed_into` | _i8 → i32_   | __2783.00__ |
| `numkong::try_dots_packed_into` | _u8 → i32_   | __2784.10__ |
| `numkong::try_dots_packed_into` | _bf16 → f32_ | __1250.80__ |
| `numkong::try_dots_packed_into` | _f16 → f32_  | __1249.70__ |
| `faer::linalg::matmul::matmul`  | _f32 → f32_  |      117.50 |
| `matrixmultiply::sgemm`         | _f32 → f32_  |      116.77 |
| `ndarray::dot`                  | _f32 → f32_  |      117.51 |
| `nalgebra::gemm`                | _f32 → f32_  |  __118.72__ |
| `numkong::try_dots_packed_into` | _f32 → f64_  |  __197.79__ |
| `numkong::try_dots_packed_into` | _e4m3 → f32_ |  __495.25__ |
| `numkong::try_dots_packed_into` | _e5m2 → f32_ |  __746.22__ |
| `numkong::try_dots_packed_into` | _e2m3 → f32_ | __1355.00__ |
| `numkong::try_dots_packed_into` | _e3m2 → f32_ |  __693.43__ |

## Python

| Library               | Precision    |       GSO/s |
| :-------------------- | :----------- | ----------: |
| `numkong.dots_packed` | _i8 → i32_   | __2621.97__ |
| `numkong.dots_packed` | _bf16 → f32_ | __1142.19__ |
| `numpy.matmul`        | _f32 → f32_  | __1854.27__ |
| `numkong.dots_packed` | _f16 → f32_  | __1134.69__ |
| `numkong.dots_packed` | _f32 → f64_  |  __194.15__ |

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
