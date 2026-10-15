# Dots Benchmarks

Packed GEMM-style matrix multiplication benchmarks comparing NumKong against faer, matrixmultiply, ndarray, nalgebra, and NumPy.
The SYRK rows multiply one matrix by its own transpose, C = A × Aᵀ, beside the GEMM rows' C = A × Bᵀ.
NumKong fills the upper triangle, faer's triangular matmul fills the lower one, NumPy hands `a @ a.T` to BLAS `syrk`, and the other competitors compute the full product.
Every SYRK row counts only the n(n+1)/2 useful pairs of one triangle times the depth, at a multiply and an add per pair and dimension like the GEMM rows, so a full product earns no credit for the mirrored half.

Speeds are in scalar operations per second: GSO/s on CPUs and TSO/s on GPUs.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong widens its outputs: i8 → i32, bf16, f16 and the FP8, FP6 and FP4 formats → f32, f32 → f64.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `faer::linalg::matmul::matmul`, `nalgebra::gemm`, `matrixmultiply::sgemm`, `matrixmultiply::dgemm`.

| NumKong                                      |       Speed | Speedup | Alternative                                |      Speed |
| :------------------------------------------- | ----------: | ------: | :----------------------------------------- | ---------: |
| `numkong::dots_packed_into`, _i8 → i32_      | 2,876 GSO/s |   28.7× | `ndarray::dot`, _i8 → i8_                  |  100 GSO/s |
| `numkong::dots_packed_into`, _u8 → u32_      | 2,855 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _bf16 → f32_    | 1,347 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _f16 → f32_     |   137 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _f32 → f64_     |  53.8 GSO/s |   0.29× | `ndarray::dot`, _f32 → f32_                |  182 GSO/s |
| `numkong::dots_packed_into`, _f64 → f64_     |  6.30 GSO/s |   0.12× | `ndarray::dot`                             | 52.7 GSO/s |
| `numkong::dots_packed_into`, _e4m3 → f32_    |   742 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _e5m2 → f32_    |   694 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _e2m3 → f32_    | 2,212 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _e3m2 → f32_    | 1,138 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _e2m1 → f32_    | 2,380 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _i4 → i32_      |  28.8 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _u4 → u32_      |   476 GSO/s |       – | –                                          |          – |
| `numkong::dots_packed_into`, _u1 → u32_      | 1,526 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _i8 → i32_   |   973 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _u8 → u32_   |   972 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _bf16 → f32_ |   424 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _f16 → f32_  |  62.9 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _f32 → f64_  |  31.3 GSO/s |   0.52× | `ndarray::dot`, _f32 → f32_, A × Aᵀ        | 60.7 GSO/s |
| `numkong::dots_symmetric_into`, _f64 → f64_  |  6.35 GSO/s |   0.14× | `faer::linalg::matmul::triangular::matmul` | 45.3 GSO/s |
| `numkong::dots_symmetric_into`, _e4m3 → f32_ |   194 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _e5m2 → f32_ |   138 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _e2m3 → f32_ |   629 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _e3m2 → f32_ |   316 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _e2m1 → f32_ |   616 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _i4 → i32_   |   312 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _u4 → u32_   |   377 GSO/s |       – | –                                          |          – |
| `numkong::dots_symmetric_into`, _u1 → u32_   | 1,165 GSO/s |       – | –                                          |          – |

> Measured October 14, 2026.

### Python

| NumKong                                |       Speed | Speedup | Alternative                         |     Speed |
| :------------------------------------- | ----------: | ------: | :---------------------------------- | --------: |
| `numkong.dots_packed`, _i8 → i32_      | 1,481 GSO/s |       – | –                                   |         – |
| `numkong.dots_packed`, _bf16 → f32_    |   691 GSO/s |       – | –                                   |         – |
| `numkong.dots_packed`, _f16 → f32_     |  61.2 GSO/s |       – | –                                   |         – |
| `numkong.dots_packed`, _f32 → f64_     |  31.5 GSO/s |   0.17× | `numpy.matmul`, _f32 → f32_         | 188 GSO/s |
| `numkong.dots_symmetric`, _i8 → i32_   |   827 GSO/s |       – | –                                   |         – |
| `numkong.dots_symmetric`, _bf16 → f32_ |   382 GSO/s |       – | –                                   |         – |
| `numkong.dots_symmetric`, _f16 → f32_  |  63.8 GSO/s |       – | –                                   |         – |
| `numkong.dots_symmetric`, _f32 → f64_  |  34.4 GSO/s |   0.23× | `numpy.matmul`, _f32 → f32_, A × Aᵀ | 149 GSO/s |

> Measured October 14, 2026.

## 148× Nvidia SM100

### Rust

| NumKong                           |       Speed | Speedup | Alternative                     |       Speed |
| :-------------------------------- | ----------: | ------: | :------------------------------ | ----------: |
| `nk_dots_packed`, _i8 → i32_      |   856 TSO/s |   0.29× | `cublasLtMatmul`                | 2,979 TSO/s |
| `nk_dots_packed`, _u8 → u32_      |   770 TSO/s |       – | –                               |           – |
| `nk_dots_packed`, _bf16 → f32_    | 1,248 TSO/s |   0.86× | `cublasLtMatmul`                | 1,455 TSO/s |
| `nk_dots_packed`, _f16 → f32_     | 1,186 TSO/s |   0.85× | `cublasLtMatmul`                | 1,391 TSO/s |
| `nk_dots_packed`, _f32 → f64_     |  14.4 TSO/s |   0.22× | `cublasLtMatmul`, _f32 → f32_   |  65.6 TSO/s |
| `nk_dots_packed`, _f64 → f64_     |  3.10 TSO/s |  0.085× | `cublasLtMatmul`                |  36.4 TSO/s |
| `nk_dots_packed`, _e4m3 → f32_    | 2,496 TSO/s |   0.91× | `cublasLtMatmul`                | 2,734 TSO/s |
| `nk_dots_packed`, _e5m2 → f32_    | 2,568 TSO/s |       – | –                               |           – |
| `nk_dots_packed`, _e2m3 → f32_    | 1,862 TSO/s |       – | –                               |           – |
| `nk_dots_packed`, _e3m2 → f32_    | 1,912 TSO/s |       – | –                               |           – |
| `nk_dots_packed`, _e2m1 → f32_    | 5,264 TSO/s |   1.06× | `cublasLtMatmul`, NVFP4         | 4,985 TSO/s |
| `nk_dots_packed`, _i4 → i32_      |   523 TSO/s |       – | –                               |           – |
| `nk_dots_packed`, _u4 → u32_      |   559 TSO/s |       – | –                               |           – |
| `nk_dots_symmetric`, _i8 → i32_   |   602 TSO/s |   0.40× | `cublasLtMatmul`, A × Aᵀ        | 1,517 TSO/s |
| `nk_dots_symmetric`, _u8 → u32_   |   558 TSO/s |       – | –                               |           – |
| `nk_dots_symmetric`, _bf16 → f32_ | 1,108 TSO/s |   1.52× | `cublasLtMatmul`, A × Aᵀ        |   729 TSO/s |
| `nk_dots_symmetric`, _f16 → f32_  | 1,011 TSO/s |   1.45× | `cublasLtMatmul`, A × Aᵀ        |   700 TSO/s |
| `nk_dots_symmetric`, _f32 → f64_  |  14.4 TSO/s |   0.26× | `cublasSsyrk`, _f32 → f32_      |  55.8 TSO/s |
| `nk_dots_symmetric`, _f64 → f64_  |  3.03 TSO/s |  0.087× | `cublasDsyrk`                   |  34.7 TSO/s |
| `nk_dots_symmetric`, _e4m3 → f32_ | 2,222 TSO/s |   1.62× | `cublasLtMatmul`, A × Aᵀ        | 1,375 TSO/s |
| `nk_dots_symmetric`, _e5m2 → f32_ | 2,312 TSO/s |       – | –                               |           – |
| `nk_dots_symmetric`, _e2m3 → f32_ | 1,181 TSO/s |       – | –                               |           – |
| `nk_dots_symmetric`, _e3m2 → f32_ | 1,182 TSO/s |       – | –                               |           – |
| `nk_dots_symmetric`, _e2m1 → f32_ | 4,721 TSO/s |   1.87× | `cublasLtMatmul`, NVFP4, A × Aᵀ | 2,525 TSO/s |
| `nk_dots_symmetric`, _i4 → i32_   |   504 TSO/s |       – | –                               |           – |
| `nk_dots_symmetric`, _u4 → u32_   |   538 TSO/s |       – | –                               |           – |

> Measured October 14, 2026.

### Python

| NumKong                                |       Speed | Speedup | Alternative                           |       Speed |
| :------------------------------------- | ----------: | ------: | :------------------------------------ | ----------: |
| `numkong.dots_packed`, _i8 → i32_      |   855 TSO/s |   0.29× | `torch._int_mm`                       | 2,993 TSO/s |
| `numkong.dots_packed`, _u8 → u32_      |   770 TSO/s |       – | –                                     |           – |
| `numkong.dots_packed`, _bf16 → f32_    | 1,254 TSO/s |   0.85× | `torch.matmul`, _bf16 → bf16_         | 1,484 TSO/s |
| `numkong.dots_packed`, _f16 → f32_     | 1,177 TSO/s |   0.84× | `torch.matmul`, _f16 → f16_           | 1,409 TSO/s |
| `numkong.dots_packed`, _f32 → f64_     |  32.7 TSO/s |   0.50× | `torch.matmul`, _f32 → f32_           |  65.6 TSO/s |
| `numkong.dots_packed`, _f64 → f64_     |  3.10 TSO/s |  0.085× | `torch.matmul`                        |  36.4 TSO/s |
| `numkong.dots_packed`, _e4m3 → f32_    | 2,486 TSO/s |   0.91× | `torch._scaled_mm`                    | 2,734 TSO/s |
| `numkong.dots_packed`, _e5m2 → f32_    | 2,589 TSO/s |       – | –                                     |           – |
| `numkong.dots_packed`, _e2m1 → f32_    | 5,268 TSO/s |       – | –                                     |           – |
| `numkong.dots_packed`, _i4 → i32_      |   523 TSO/s |       – | –                                     |           – |
| `numkong.dots_packed`, _u4 → u32_      |   559 TSO/s |       – | –                                     |           – |
| `numkong.dots_symmetric`, _i8 → i32_   |   602 TSO/s |   0.40× | `torch._int_mm`, A × Aᵀ               | 1,509 TSO/s |
| `numkong.dots_symmetric`, _u8 → u32_   |   558 TSO/s |       – | –                                     |           – |
| `numkong.dots_symmetric`, _bf16 → f32_ | 1,105 TSO/s |   1.49× | `torch.matmul`, _bf16 → bf16_, A × Aᵀ |   743 TSO/s |
| `numkong.dots_symmetric`, _f16 → f32_  | 1,012 TSO/s |   1.42× | `cupy.matmul`, _f16 → f16_, A × Aᵀ    |   711 TSO/s |
| `numkong.dots_symmetric`, _f32 → f64_  |  31.3 TSO/s |   0.95× | `torch.matmul`, _f32 → f32_, A × Aᵀ   |  32.8 TSO/s |
| `numkong.dots_symmetric`, _f64 → f64_  |  3.03 TSO/s |   0.17× | `torch.matmul`, A × Aᵀ                |  18.2 TSO/s |
| `numkong.dots_symmetric`, _e4m3 → f32_ | 2,227 TSO/s |   1.62× | `torch._scaled_mm`, A × Aᵀ            | 1,371 TSO/s |
| `numkong.dots_symmetric`, _e5m2 → f32_ | 2,315 TSO/s |       – | –                                     |           – |
| `numkong.dots_symmetric`, _e2m1 → f32_ | 4,710 TSO/s |       – | –                                     |           – |
| `numkong.dots_symmetric`, _i4 → i32_   |   503 TSO/s |       – | –                                     |           – |
| `numkong.dots_symmetric`, _u4 → u32_   |   538 TSO/s |       – | –                                     |           – |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `faer::linalg::matmul::matmul`, `matrixmultiply::sgemm`.

| NumKong                                   |       Speed | Speedup | Alternative                   |     Speed |
| :---------------------------------------- | ----------: | ------: | :---------------------------- | --------: |
| `numkong::dots_packed_into`, _i8 → i32_   | 2,783 GSO/s |       – | `ndarray::dot`, _i8 → i8_     |         ⋯ |
| `numkong::dots_packed_into`, _u8 → u32_   | 2,784 GSO/s |       – | –                             |         – |
| `numkong::dots_packed_into`, _bf16 → f32_ | 1,251 GSO/s |       – | –                             |         – |
| `numkong::dots_packed_into`, _f16 → f32_  | 1,250 GSO/s |       – | –                             |         – |
| `numkong::dots_packed_into`, _f32 → f64_  |   198 GSO/s |   1.67× | `nalgebra::gemm`, _f32 → f32_ | 119 GSO/s |
| `numkong::dots_packed_into`, _e4m3 → f32_ |   495 GSO/s |       – | –                             |         – |
| `numkong::dots_packed_into`, _e5m2 → f32_ |   746 GSO/s |       – | –                             |         – |
| `numkong::dots_packed_into`, _e2m3 → f32_ | 1,355 GSO/s |       – | –                             |         – |
| `numkong::dots_packed_into`, _e3m2 → f32_ |   693 GSO/s |       – | –                             |         – |

> Measured March 22, 2026.

### Python

| NumKong                             |       Speed | Speedup | Alternative                 |       Speed |
| :---------------------------------- | ----------: | ------: | :-------------------------- | ----------: |
| `numkong.dots_packed`, _i8 → i32_   | 2,622 GSO/s |       – | –                           |           – |
| `numkong.dots_packed`, _bf16 → f32_ | 1,142 GSO/s |       – | –                           |           – |
| `numkong.dots_packed`, _f16 → f32_  | 1,135 GSO/s |       – | –                           |           – |
| `numkong.dots_packed`, _f32 → f64_  |   194 GSO/s |   0.10× | `numpy.matmul`, _f32 → f32_ | 1,854 GSO/s |

> Measured March 22, 2026.

## Run It

### Rust

```bash
# Default 1024×128×1536 workload
cargo bench --bench bench_dots --features bench_dots

# Smaller 512×512×512 workload
NUMWARS_DIMS_WIDTH=512 NUMWARS_DIMS_HEIGHT=512 NUMWARS_DIMS_DEPTH=512 \
cargo bench --bench bench_dots --features bench_dots

# Focus on float32
NUMWARS_FILTER="dots/f32" \
cargo bench --bench bench_dots --features bench_dots

# Only the SYRK rows, 1024×1024 Gram matrices at 1536 dimensions
NUMWARS_FILTER="dots/symmetric" \
cargo bench --bench bench_dots --features bench_dots

# Only the GPU rows, on device zero at the 16384×16384×16384 shape above
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda NUMWARS_SAMPLE_SIZE=10 \
NUMWARS_DIMS_HEIGHT=16384 NUMWARS_DIMS_WIDTH=16384 NUMWARS_DIMS_DEPTH=16384 \
cargo bench --bench bench_dots --features "bench_dots cuda"
```

### Python

```bash
# Default 1024×128×1536 workload, f32 only
NUMWARS_FILTER='dots/numpy/f32' python dots/bench.py

# Only the SYRK rows, 1024×1024 Gram matrices at 1536 dimensions
NUMWARS_FILTER='dots/symmetric' python dots/bench.py

# Only the GPU rows, on device zero at the 16384×16384×16384 shape above
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda \
NUMWARS_DIMS_HEIGHT=16384 NUMWARS_DIMS_WIDTH=16384 NUMWARS_DIMS_DEPTH=16384 python dots/bench.py
```
