# Similarities Benchmarks

All-pairs distance matrix benchmarks comparing NumKong packed kernels against ndarray and nalgebra.
The SYRK rows score one matrix against itself, A × Aᵀ, where NumKong's `*_symmetric` kernels fill the upper triangle.
In Rust, ndarray and nalgebra compute the full Gram matrix and apply the distance formula to its upper triangle, reading the norms off its diagonal, and a serial `count_ones` loop over 64-bit words is the bit-set baseline.
In Python, SciPy's `pdist` returns the condensed upper triangle without the diagonal, and its Hamming distance is the fraction of differing bits rather than their count.
Every SYRK row counts only the n(n+1)/2 useful pairs of one triangle times the depth, so a full product earns no credit for the mirrored half.
The packed rows count all n² pairs times the depth, and the Rust ones derive their GSO/s from Criterion's time per call.

Speeds are in scalar operations per second: GSO/s on CPUs and TSO/s on GPUs.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong widens its outputs: i8, u8, i4, u4, bf16, f16 and the FP8, FP6 and FP4 formats → f32, f32 → f64.
The Rust ndarray and nalgebra alternatives are named by the matrix product they run before the distance epilogue, and `baseline_*` are the serial `count_ones` loops of `bench.rs`.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `nalgebra::DMatrix::mul`.

| NumKong                                                    |       Speed | Speedup | Alternative                         |      Speed |
| :--------------------------------------------------------- | ----------: | ------: | :---------------------------------- | ---------: |
| `numkong::Tensor::angulars_packed`, _i8 → f32_             |   667 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::angulars_packed`, _u8 → f32_             |   671 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::angulars_packed`, _bf16 → f32_           |   389 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::angulars_packed`, _f32 → f64_            |  24.4 GSO/s |   0.26× | `ndarray::dot`, _f32 → f32_         | 95.2 GSO/s |
| `numkong::Tensor::angulars_packed`, _f64 → f64_            |  5.69 GSO/s |   0.12× | `ndarray::dot`                      | 46.5 GSO/s |
| `numkong::Tensor::angulars_symmetric_into`, _i8 → f32_     |   338 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::angulars_symmetric_into`, _u8 → f32_     |   340 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::angulars_symmetric_into`, _bf16 → f32_   |   171 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::angulars_symmetric_into`, _f32 → f64_    |  16.7 GSO/s |   0.33× | `ndarray::dot`, _f32 → f32_, A × Aᵀ | 50.5 GSO/s |
| `numkong::Tensor::angulars_symmetric_into`, _f64 → f64_    |  5.51 GSO/s |   0.23× | `ndarray::dot`, A × Aᵀ              | 24.0 GSO/s |
| `numkong::Tensor::euclideans_packed`, _i8 → f32_           |   712 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::euclideans_packed`, _u8 → f32_           |   709 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::euclideans_packed`, _bf16 → f32_         |   395 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::euclideans_packed`, _f32 → f64_          |  25.1 GSO/s |   0.26× | `ndarray::dot`, _f32 → f32_         | 96.9 GSO/s |
| `numkong::Tensor::euclideans_packed`, _f64 → f64_          |  5.68 GSO/s |   0.12× | `ndarray::dot`                      | 47.1 GSO/s |
| `numkong::Tensor::euclideans_symmetric_into`, _i8 → f32_   |   348 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::euclideans_symmetric_into`, _u8 → f32_   |   350 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::euclideans_symmetric_into`, _bf16 → f32_ |   176 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::euclideans_symmetric_into`, _f32 → f64_  |  16.6 GSO/s |   0.32× | `ndarray::dot`, _f32 → f32_, A × Aᵀ | 51.3 GSO/s |
| `numkong::Tensor::euclideans_symmetric_into`, _f64 → f64_  |  5.55 GSO/s |   0.23× | `ndarray::dot`, A × Aᵀ              | 24.4 GSO/s |
| `numkong::Tensor::hammings_packed`, _u1x8_                 |   617 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::hammings_symmetric_into`, _u1x8_         |   496 GSO/s |   1.50× | `baseline_hammings_symmetric`       |  330 GSO/s |
| `numkong::Tensor::jaccards_packed`, _u1x8_                 |   601 GSO/s |       – | –                                   |          – |
| `numkong::Tensor::jaccards_symmetric_into`, _u1x8_         |   474 GSO/s |   2.63× | `baseline_jaccards_symmetric`       |  180 GSO/s |

> Measured October 14, 2026.

### Python

| NumKong                                      |       Speed | Speedup | Alternative                                         |      Speed |
| :------------------------------------------- | ----------: | ------: | :-------------------------------------------------- | ---------: |
| `numkong.euclideans_packed`, _u8 → f32_      |   550 GSO/s |       – | –                                                   |          – |
| `numkong.euclideans_packed`, _i8 → f32_      |   545 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_packed`, _i8 → f32_        |   522 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_packed`, _u8 → f32_        |   523 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_packed`, _bf16 → f32_      |   297 GSO/s |       – | –                                                   |          – |
| `numkong.euclideans_packed`, _bf16 → f32_    |   295 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_packed`, _f32 → f64_       |  15.4 GSO/s |   6.00× | `scipy.spatial.distance.cdist`, cosine              | 2.57 GSO/s |
| `numkong.euclideans_packed`, _f32 → f64_     |  15.6 GSO/s |   4.46× | `scipy.spatial.distance.cdist`, euclidean           | 3.49 GSO/s |
| `numkong.euclideans_symmetric`, _u8 → f32_   |   195 GSO/s |       – | –                                                   |          – |
| `numkong.euclideans_symmetric`, _i8 → f32_   |   195 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_symmetric`, _i8 → f32_     |   192 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_symmetric`, _u8 → f32_     |   193 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_symmetric`, _bf16 → f32_   |  89.7 GSO/s |       – | –                                                   |          – |
| `numkong.euclideans_symmetric`, _bf16 → f32_ |  90.1 GSO/s |       – | –                                                   |          – |
| `numkong.angulars_symmetric`, _f32 → f64_    |  14.5 GSO/s |   5.56× | `scipy.spatial.distance.pdist`, cosine              | 2.61 GSO/s |
| `numkong.euclideans_symmetric`, _f32 → f64_  |  14.5 GSO/s |   4.23× | `scipy.spatial.distance.pdist`, euclidean           | 3.44 GSO/s |
| `numkong.hammings_symmetric`, _u1 → u32_     |   246 GSO/s |   69.0× | `scipy.spatial.distance.pdist`, _bool → f64_        | 3.57 GSO/s |
| `numkong.jaccards_symmetric`, _u1 → f32_     |   240 GSO/s |    349× | `scipy.spatial.distance.pdist`, _bool → f64_        | 0.69 GSO/s |

> Measured October 14, 2026.

## 148× Nvidia SM100

### Rust

| NumKong                                      |      Speed | Speedup | Alternative                                   |      Speed |
| :------------------------------------------- | ---------: | ------: | :-------------------------------------------- | ---------: |
| `nk_angulars_packed`, _i8 → f32_             | 81.8 TSO/s |   0.23× | `cublasLtMatmul`                              |  363 TSO/s |
| `nk_angulars_packed`, _u8 → f32_             |  111 TSO/s |       – | –                                             |          – |
| `nk_angulars_packed`, _bf16 → f32_           |  265 TSO/s |   1.13× | `cublasLtMatmul`                              |  234 TSO/s |
| `nk_angulars_packed`, _f16 → f32_            |  262 TSO/s |   0.89× | `cublasLtMatmul`                              |  295 TSO/s |
| `nk_angulars_packed`, _f32 → f64_            | 7.07 TSO/s |   0.24× | `cublasLtMatmul`, _f32 → f32_                 | 29.9 TSO/s |
| `nk_angulars_packed`, _f64 → f64_            | 1.52 TSO/s |  0.088× | `cublasLtMatmul`                              | 17.4 TSO/s |
| `nk_angulars_packed`, _e4m3 → f32_           |  238 TSO/s |   0.67× | `cublasLtMatmul`                              |  357 TSO/s |
| `nk_angulars_packed`, _e5m2 → f32_           |  284 TSO/s |       – | –                                             |          – |
| `nk_angulars_packed`, _e2m3 → f32_           |  244 TSO/s |       – | –                                             |          – |
| `nk_angulars_packed`, _e3m2 → f32_           |  258 TSO/s |       – | –                                             |          – |
| `nk_angulars_packed`, _e2m1 → f32_           |  298 TSO/s |       – | –                                             |          – |
| `nk_angulars_packed`, _i4 → f32_             | 81.3 TSO/s |       – | –                                             |          – |
| `nk_angulars_packed`, _u4 → f32_             |  111 TSO/s |       – | –                                             |          – |
| `nk_angulars_symmetric`, _i8 → f32_          | 72.3 TSO/s |   0.39× | `cublasLtMatmul`, A × Aᵀ                      |  185 TSO/s |
| `nk_angulars_symmetric`, _u8 → f32_          | 92.0 TSO/s |       – | –                                             |          – |
| `nk_angulars_symmetric`, _bf16 → f32_        |  173 TSO/s |   1.12× | `cublasLtMatmul`, A × Aᵀ                      |  154 TSO/s |
| `nk_angulars_symmetric`, _f16 → f32_         |  176 TSO/s |   1.18× | `cublasLtMatmul`, A × Aᵀ                      |  149 TSO/s |
| `nk_angulars_symmetric`, _f32 → f64_         | 6.77 TSO/s |   0.45× | `cublasLtMatmul`, _f32 → f32_, A × Aᵀ         | 15.0 TSO/s |
| `nk_angulars_symmetric`, _f64 → f64_         | 1.45 TSO/s |   0.17× | `cublasLtMatmul`, A × Aᵀ                      | 8.69 TSO/s |
| `nk_angulars_symmetric`, _e4m3 → f32_        |  104 TSO/s |   0.58× | `cublasLtMatmul`, A × Aᵀ                      |  181 TSO/s |
| `nk_angulars_symmetric`, _e5m2 → f32_        |  183 TSO/s |       – | –                                             |          – |
| `nk_angulars_symmetric`, _e2m3 → f32_        |  137 TSO/s |       – | –                                             |          – |
| `nk_angulars_symmetric`, _e3m2 → f32_        |  129 TSO/s |       – | –                                             |          – |
| `nk_angulars_symmetric`, _e2m1 → f32_        |  195 TSO/s |       – | –                                             |          – |
| `nk_angulars_symmetric`, _i4 → f32_          | 72.6 TSO/s |       – | –                                             |          – |
| `nk_angulars_symmetric`, _u4 → f32_          | 98.0 TSO/s |       – | –                                             |          – |
| `nk_euclideans_packed`, _i8 → f32_           |  208 TSO/s |   0.63× | `cublasLtMatmul`                              |  330 TSO/s |
| `nk_euclideans_packed`, _u8 → f32_           |  194 TSO/s |       – | –                                             |          – |
| `nk_euclideans_packed`, _bf16 → f32_         |  290 TSO/s |   1.04× | `cublasLtMatmul`                              |  278 TSO/s |
| `nk_euclideans_packed`, _f16 → f32_          |  273 TSO/s |   1.02× | `cublasLtMatmul`                              |  269 TSO/s |
| `nk_euclideans_packed`, _f32 → f64_          | 7.13 TSO/s |   0.24× | `cublasLtMatmul`, _f32 → f32_                 | 29.7 TSO/s |
| `nk_euclideans_packed`, _f64 → f64_          | 1.53 TSO/s |  0.088× | `cublasLtMatmul`                              | 17.3 TSO/s |
| `nk_euclideans_packed`, _e4m3 → f32_         |  275 TSO/s |   0.85× | `cublasLtMatmul`                              |  325 TSO/s |
| `nk_euclideans_packed`, _e5m2 → f32_         |  323 TSO/s |       – | –                                             |          – |
| `nk_euclideans_packed`, _e2m3 → f32_         |  283 TSO/s |       – | –                                             |          – |
| `nk_euclideans_packed`, _e3m2 → f32_         |  292 TSO/s |       – | –                                             |          – |
| `nk_euclideans_packed`, _e2m1 → f32_         |  341 TSO/s |       – | –                                             |          – |
| `nk_euclideans_packed`, _i4 → f32_           |  204 TSO/s |       – | –                                             |          – |
| `nk_euclideans_packed`, _u4 → f32_           |  209 TSO/s |       – | –                                             |          – |
| `nk_euclideans_symmetric`, _i8 → f32_        |  143 TSO/s |   0.86× | `cublasLtMatmul`, A × Aᵀ                      |  167 TSO/s |
| `nk_euclideans_symmetric`, _u8 → f32_        |  137 TSO/s |       – | –                                             |          – |
| `nk_euclideans_symmetric`, _bf16 → f32_      |  216 TSO/s |   1.55× | `cublasLtMatmul`, A × Aᵀ                      |  140 TSO/s |
| `nk_euclideans_symmetric`, _f16 → f32_       |  221 TSO/s |   1.63× | `cublasLtMatmul`, A × Aᵀ                      |  136 TSO/s |
| `nk_euclideans_symmetric`, _f32 → f64_       | 6.79 TSO/s |   0.46× | `cublasLtMatmul`, _f32 → f32_, A × Aᵀ         | 14.8 TSO/s |
| `nk_euclideans_symmetric`, _f64 → f64_       | 1.45 TSO/s |   0.17× | `cublasLtMatmul`, A × Aᵀ                      | 8.66 TSO/s |
| `nk_euclideans_symmetric`, _e4m3 → f32_      |  110 TSO/s |   0.67× | `cublasLtMatmul`, A × Aᵀ                      |  164 TSO/s |
| `nk_euclideans_symmetric`, _e5m2 → f32_      |  234 TSO/s |       – | –                                             |          – |
| `nk_euclideans_symmetric`, _e2m3 → f32_      |  152 TSO/s |       – | –                                             |          – |
| `nk_euclideans_symmetric`, _e3m2 → f32_      |  145 TSO/s |       – | –                                             |          – |
| `nk_euclideans_symmetric`, _e2m1 → f32_      |  255 TSO/s |       – | –                                             |          – |
| `nk_euclideans_symmetric`, _i4 → f32_        |  146 TSO/s |       – | –                                             |          – |
| `nk_euclideans_symmetric`, _u4 → f32_        |  150 TSO/s |       – | –                                             |          – |

> Measured October 14, 2026.

### Python

| NumKong                                      |      Speed | Speedup | Alternative                                   |      Speed |
| :------------------------------------------- | ---------: | ------: | :-------------------------------------------- | ---------: |
| `numkong.angulars_packed`, _i8 → f32_        | 81.8 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_packed`, _i8 → f32_      |  208 TSO/s |       – | –                                             |          – |
| `numkong.angulars_packed`, _u8 → f32_        |  111 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_packed`, _u8 → f32_      |  194 TSO/s |       – | –                                             |          – |
| `numkong.angulars_packed`, _i4 → f32_        | 81.6 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_packed`, _i4 → f32_      |  204 TSO/s |       – | –                                             |          – |
| `numkong.angulars_packed`, _u4 → f32_        |  111 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_packed`, _u4 → f32_      |  209 TSO/s |       – | –                                             |          – |
| `numkong.angulars_packed`, _bf16 → f32_      |  266 TSO/s |   1.02× | `cupy.matmul`                                 |  260 TSO/s |
| `numkong.euclideans_packed`, _bf16 → f32_    |  286 TSO/s |   1.10× | `cupy.matmul`                                 |  261 TSO/s |
| `numkong.angulars_packed`, _f16 → f32_       |  262 TSO/s |   1.02× | `cupy.matmul`                                 |  258 TSO/s |
| `numkong.euclideans_packed`, _f16 → f32_     |  274 TSO/s |   1.06× | `cupy.matmul`                                 |  258 TSO/s |
| `numkong.angulars_packed`, _e4m3 → f32_      |  238 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_packed`, _e4m3 → f32_    |  275 TSO/s |       – | –                                             |          – |
| `numkong.angulars_packed`, _e5m2 → f32_      |  283 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_packed`, _e5m2 → f32_    |  323 TSO/s |       – | –                                             |          – |
| `numkong.angulars_packed`, _e2m1 → f32_      |  298 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_packed`, _e2m1 → f32_    |  341 TSO/s |       – | –                                             |          – |
| `numkong.angulars_packed`, _f32 → f64_       | 13.9 TSO/s |   0.47× | `cupy.matmul`, _f32 → f32_                    | 29.2 TSO/s |
| `numkong.euclideans_packed`, _f32 → f64_     | 14.1 TSO/s |   0.48× | `torch.cdist`, _f32 → f32_                    | 29.5 TSO/s |
| `numkong.angulars_packed`, _f64 → f64_       | 1.52 TSO/s |  0.089× | `cupy.matmul`                                 | 17.2 TSO/s |
| `numkong.euclideans_packed`, _f64 → f64_     | 1.53 TSO/s |  0.089× | `cupy.matmul`                                 | 17.2 TSO/s |
| `numkong.angulars_symmetric`, _i8 → f32_     | 72.3 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_symmetric`, _i8 → f32_   |  143 TSO/s |       – | –                                             |          – |
| `numkong.angulars_symmetric`, _u8 → f32_     | 91.9 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_symmetric`, _u8 → f32_   |  137 TSO/s |       – | –                                             |          – |
| `numkong.angulars_symmetric`, _i4 → f32_     | 72.5 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_symmetric`, _i4 → f32_   |  145 TSO/s |       – | –                                             |          – |
| `numkong.angulars_symmetric`, _u4 → f32_     | 97.9 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_symmetric`, _u4 → f32_   |  150 TSO/s |       – | –                                             |          – |
| `numkong.angulars_symmetric`, _bf16 → f32_   |  173 TSO/s |   1.32× | `cupy.matmul`, A × Aᵀ                         |  130 TSO/s |
| `numkong.euclideans_symmetric`, _bf16 → f32_ |  215 TSO/s |   1.65× | `cupy.matmul`, A × Aᵀ                         |  130 TSO/s |
| `numkong.angulars_symmetric`, _f16 → f32_    |  176 TSO/s |   1.36× | `cupy.matmul`, A × Aᵀ                         |  129 TSO/s |
| `numkong.euclideans_symmetric`, _f16 → f32_  |  221 TSO/s |   1.73× | `cupy.matmul`, A × Aᵀ                         |  128 TSO/s |
| `numkong.angulars_symmetric`, _e4m3 → f32_   |  104 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_symmetric`, _e4m3 → f32_ |  110 TSO/s |       – | –                                             |          – |
| `numkong.angulars_symmetric`, _e5m2 → f32_   |  183 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_symmetric`, _e5m2 → f32_ |  234 TSO/s |       – | –                                             |          – |
| `numkong.angulars_symmetric`, _e2m1 → f32_   |  195 TSO/s |       – | –                                             |          – |
| `numkong.euclideans_symmetric`, _e2m1 → f32_ |  255 TSO/s |       – | –                                             |          – |
| `numkong.angulars_symmetric`, _f32 → f64_    | 12.0 TSO/s |   0.82× | `cupy.matmul`, _f32 → f32_, A × Aᵀ            | 14.6 TSO/s |
| `numkong.euclideans_symmetric`, _f32 → f64_  | 12.1 TSO/s |   0.82× | `torch.cdist`, _f32 → f32_, A × Aᵀ            | 14.8 TSO/s |
| `numkong.angulars_symmetric`, _f64 → f64_    | 1.45 TSO/s |   0.17× | `cupy.matmul`, A × Aᵀ                         | 8.60 TSO/s |
| `numkong.euclideans_symmetric`, _f64 → f64_  | 1.45 TSO/s |   0.17× | `cupy.matmul`, A × Aᵀ                         | 8.60 TSO/s |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `nalgebra::DMatrix::mul`.

| NumKong                                            |       Speed | Speedup | Alternative                 |      Speed |
| :------------------------------------------------- | ----------: | ------: | :-------------------------- | ---------: |
| `numkong::Tensor::angulars_packed`, _i8 → f32_     |   830 GSO/s |       – | –                           |          – |
| `numkong::Tensor::angulars_packed`, _u8 → f32_     |   830 GSO/s |       – | –                           |          – |
| `numkong::Tensor::angulars_packed`, _bf16 → f32_   |   502 GSO/s |       – | –                           |          – |
| `numkong::Tensor::angulars_packed`, _f32 → f64_    |  92.5 GSO/s |   1.62× | `ndarray::dot`, _f32 → f32_ | 57.0 GSO/s |
| `numkong::Tensor::angulars_packed`, _f64 → f64_    |  22.8 GSO/s |   0.79× | `ndarray::dot`              | 28.8 GSO/s |
| `numkong::Tensor::euclideans_packed`, _i8 → f32_   |   888 GSO/s |       – | –                           |          – |
| `numkong::Tensor::euclideans_packed`, _u8 → f32_   |   889 GSO/s |       – | –                           |          – |
| `numkong::Tensor::euclideans_packed`, _bf16 → f32_ |   524 GSO/s |       – | –                           |          – |
| `numkong::Tensor::euclideans_packed`, _f32 → f64_  |  92.9 GSO/s |   1.61× | `ndarray::dot`, _f32 → f32_ | 57.6 GSO/s |
| `numkong::Tensor::euclideans_packed`, _f64 → f64_  |  22.9 GSO/s |   0.79× | `ndarray::dot`              | 28.8 GSO/s |
| `numkong::Tensor::hammings_packed`, _u1x8_         | 9,821 GSO/s |       – | –                           |          – |
| `numkong::Tensor::jaccards_packed`, _u1x8_         | 3,173 GSO/s |       – | –                           |          – |

> Measured March 22, 2026.

### Python

| NumKong                                  |      Speed | Speedup | Alternative                               |      Speed |
| :--------------------------------------- | ---------: | ------: | :---------------------------------------- | ---------: |
| `numkong.euclideans_packed`, _u8 → f32_  |  426 GSO/s |       – | –                                         |          – |
| `numkong.euclideans_packed`, _i8 → f32_  |  409 GSO/s |       – | –                                         |          – |
| `numkong.angulars_packed`, _i8 → f32_    |  387 GSO/s |       – | –                                         |          – |
| `numkong.angulars_packed`, _u8 → f32_    |  364 GSO/s |       – | –                                         |          – |
| `numkong.angulars_packed`, _f32 → f64_   | 79.3 GSO/s |   61.0× | `scipy.spatial.distance.cdist`, cosine    | 1.30 GSO/s |
| `numkong.euclideans_packed`, _f32 → f64_ | 53.0 GSO/s |   10.4× | `scipy.spatial.distance.cdist`, euclidean | 5.09 GSO/s |

> Measured March 22, 2026.

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

# Only the GPU rows, on device zero at 16384×16384 pairs of 1536 dimensions
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda NUMWARS_SAMPLE_SIZE=10 NUMWARS_BATCH_PER_CORE=16384 NUMWARS_DIMS=1536 \
cargo bench --bench bench_similarities --features "bench_similarities cuda"
```

### Python

```bash
# Run the Python suite
python similarities/bench.py

# Only the SYRK rows
NUMWARS_FILTER='similarities/symmetric' python similarities/bench.py

# Only the GPU rows, on device zero at 16384×16384 pairs of 1536 dimensions
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda NUMWARS_BATCH_PER_CORE=16384 NUMWARS_DIMS=1536 python similarities/bench.py
```
