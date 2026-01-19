# Matrix Multiplication (GEMM) Benchmarks

General Matrix Multiply (GEMM) benchmarking module comparing NumKong against BLAS implementations and deep learning frameworks.

## Overview

This module benchmarks __matrix multiplication__ operations computing `C = A @ B.T` (NT layout, No-Transpose × Transpose), which is the standard convention for:

- Neural network inference (weight × input)
- Attention mechanisms (Q @ K.T)
- NVIDIA cuBLAS GEMM operations
- Batch dot products

## Matrix Layouts

### NT Layout (NumKong, NVIDIA convention)

```
C (m×n) = A (m×k) @ B.T (n×k)
```

- __A__: m rows × k columns (row-major)
- __B__: n rows × k columns (row-major, transposed on-the-fly)
- __C__: m rows × n columns (row-major)

This matches PyTorch `A @ B.T` and NumPy `A @ B.T`.

### Why NT Layout?

1. __Cache efficiency__: B is accessed row-wise (sequential)
2. __SIMD friendly__: Inner loop processes contiguous memory
3. __Neural networks__: Weights are stored as (out_features, in_features)
4. __GPU kernels__: NVIDIA Tensor Cores use NT layout

## Supported Data Types

### Floating Point

- __f64__: 64-bit double precision
- __f32__: 32-bit single precision
- __f16__: 16-bit half precision (fast on modern GPUs)
- __bf16__: 16-bit Brain Float (TPU/GPU optimized)
- __e4m3__: 8-bit float (ML accelerators)
- __e5m2__: 8-bit float (ML accelerators)

### Integer

- __i8__: 8-bit signed integers (quantized neural networks)
- __i4__: 4-bit integers (extreme quantization)

## Usage

### Rust Benchmarks

```bash
# Run all GEMM benchmarks
cargo bench --features bench_dots --bench bench_dots

# Specific matrix sizes
NUMWARS_MATRIX_SIZES=512,1024,2048 cargo bench --features bench_dots

# Specific data types
NUMWARS_DTYPE=f32,f16 cargo bench --features bench_dots

# Multi-threaded benchmarks
NUMWARS_THREADS=8 NUMWARS_PARALLEL=true cargo bench --features bench_dots

# Filter by name
NUMWARS_FILTER=f32 cargo bench --features bench_dots
```

### Python Benchmarks

```bash
# Run all GEMM benchmarks
python dots/bench.py

# Specific matrix sizes
python dots/bench.py --sizes 512,1024,2048

# Specific data types
python dots/bench.py --dtype f32,f64

# Single-threaded (for fair comparison)
python dots/bench.py --threads 1

# Multi-threaded
python dots/bench.py --threads 8
```

## Environment Variables

### Common

- `NUMWARS_DTYPE`: Data type filter (default: "f32")
- `NUMWARS_FILTER`: Benchmark name regex filter
- `NUMWARS_WARMUP_SECONDS`: Warmup time in seconds (default: 3.0) (was: NUMWARS_WARMUP)
- `NUMWARS_PROFILE_SECONDS`: Measurement time in seconds (default: 10.0) (was: NUMWARS_TIME_LIMIT)

### GEMM-Specific

- `NUMWARS_MATRIX_SIZES`: Comma-separated sizes (default: "1024,2048,4096")
- `NUMWARS_THREADS`: Thread count (default: num_cpus)
- `NUMWARS_PARALLEL`: Enable parallel benchmarks (default: true)

## Performance Metrics

All benchmarks report:

- __Operations/sec__: Auto-scaling format (KiloOps/s, MegaOps/s, GigaOps/s, TeraOps/s)
  - Formula: `(2 × m × n × k) / duration`
  - For matmul: 2 ops per element (multiply + add)
  - Automatically selects appropriate scale based on magnitude

- __GB/s__: Memory throughput
  - Formula: `(m×k + n×k + m×n) × dtype_size / duration / 1e9`

- __Time__: Latency per operation

### Example Calculation (f32, 1024×1024)

- Operations: `2 × 1024 × 1024 × 1024 = 2,147,483,648` (~2.15 billion)
- Memory: `(1024×1024 + 1024×1024 + 1024×1024) × 4 bytes = 12 MB`
- If duration = 0.01s:
  - Operations/sec = 2.15 billion / 0.01s = 215 GigaOps/s
  - GB/s = 12 / 0.01 = 1,200 GB/s

## Matrix Sizes

Default benchmark sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384

### Size Categories

- __Tiny (64-256)__: L1 cache fits entirely, ~100-500 GigaOps/s
- __Small (512-1024)__: L2 cache, ~200-800 GigaOps/s
- __Medium (2048-4096)__: L3 cache, ~300-1000 GigaOps/s
- __Large (8192+)__: DRAM bound, depends on memory bandwidth

For square matrices (m=n=k=size):
- 1024×1024: 4 MB per matrix (f32), 12 MB total
- 4096×4096: 64 MB per matrix (f32), 192 MB total

## Example Results

Results will be added here after running benchmarks. Example format:

### Single-Threaded (f32)

| Size | NumKong (GigaOps/s) | NumPy/MKL (GigaOps/s) | OpenBLAS (GigaOps/s) | Speedup |
|-----:|--------------------:|----------------------:|---------------------:|--------:|
| 512 | 42.3 | 38.5 | 35.2 | 1.10× |
| 1024 | 185.7 | 172.3 | 158.9 | 1.08× |
| 2048 | 412.8 | 398.1 | 365.4 | 1.04× |
| 4096 | 523.4 | 515.2 | 478.6 | 1.02× |

### Multi-Threaded (f32, 8 threads)

| Size | NumKong (GigaOps/s) | NumPy/MKL (GigaOps/s) | PyTorch (GigaOps/s) | Speedup |
|-----:|--------------------:|----------------------:|--------------------:|--------:|
| 1024 | 892.4 | 1024.5 | 985.3 | 0.87× |
| 2048 | 2145.8 | 2312.1 | 2198.7 | 0.93× |
| 4096 | 3254.7 | 3512.8 | 3401.2 | 0.93× |

## Parallelism Strategy

NumKong does __not__ have built-in multi-threading yet. For parallel benchmarks:

- __Rust__: Use Fork Union thread pool at benchmark level
  - Split matrix rows across threads
  - Each thread computes subset of output rows
  - Demonstrated in `bench_dots.rs`

- __Python__: Compare against multi-threaded BLAS (MKL/OpenBLAS)
  - NumPy automatically uses all cores if available
  - Set `OMP_NUM_THREADS=1` for single-threaded comparison

## Competitors

### Rust

- __Baseline__: Naive triple-loop implementation
- __ndarray__: `ArrayBase::dot()` with BLAS backend
- __nalgebra__: `Matrix::mul()` optimized matmul
- __faer__: Cutting-edge Rust linear algebra library
- __BLAS bindings__: `blas-src`, `openblas-src`, `intel-mkl-src`

### Python

- __NumPy__: `A @ B.T` backed by OpenBLAS/MKL/BLIS
  - MKL (Intel): Highly optimized for Intel CPUs
  - OpenBLAS: Open-source, competitive performance
  - BLIS: AMD's optimized BLAS

- __PyTorch__ (optional): `A @ B.T` with MKL backend
  - GPU support (CUDA, ROCm)
  - TorchScript compilation

- __JAX__ (optional): `jax.numpy.dot(A, B.T)` with XLA
  - JIT compilation
  - TPU support

- __TensorFlow__ (optional): `tf.matmul(A, B, transpose_b=True)`
  - GPU/TPU support
  - Graph optimization

## Hardware Considerations

### CPU Architecture

Modern CPUs have specialized GEMM instructions:
- __Intel AVX-512__: VNNI (Vector Neural Network Instructions) for int8
- __ARM NEON__: SDOT/UDOT for int8 dot products
- __Apple AMX__: Matrix coprocessor (hidden from user)

Peak operations/sec estimation:
```
GigaOps/s = cores × frequency × SIMD_width × FMA_throughput
```

Example (Intel i9-12900K, 1 core, f32):
- 8 AVX-512 lanes × 2 (FMA) × 5.0 GHz = 80 GigaOps/s per core
- 16 cores × 80 = 1,280 GigaOps/s theoretical peak

### Memory Bandwidth

For large matrices, GEMM becomes memory-bound:
- DDR4-3200: ~25 GB/s per channel, ~50 GB/s dual-channel
- DDR5-5600: ~44 GB/s per channel, ~88 GB/s dual-channel
- LPDDR5-6400: ~51 GB/s per channel

Arithmetic intensity (ops per byte):
```
Intensity = (2×m×n×k) / ((m×k + n×k + m×n) × dtype_size)
```

For square 1024×1024 f32:
- Intensity = 2×1024³ / (3×1024²×4) = 170.7 ops/byte

This is **very high**, so GEMM is compute-bound (good for optimization).

## Optimization Techniques

### Cache Blocking (Tiling)

Split matrices into blocks that fit in cache:
```rust
for i in (0..m).step_by(BLOCK_SIZE) {
    for j in (0..n).step_by(BLOCK_SIZE) {
        for k in (0..k_dim).step_by(BLOCK_SIZE) {
            // Compute C[i..i+BLOCK, j..j+BLOCK]
        }
    }
}
```

Optimal block sizes:
- L1 cache (32 KB): 64×64 f32 blocks
- L2 cache (256 KB): 128×128 f32 blocks
- L3 cache (16 MB): 512×512 f32 blocks

### Kernel Micro-Optimizations

- __Register blocking__: Keep sub-blocks in registers (4×4, 8×8)
- __Loop unrolling__: Reduce loop overhead
- __SIMD intrinsics__: Explicit vectorization
- __FMA instructions__: Fused multiply-add (2 ops in 1 instruction)

### Data Layout

- __Row-major (C, NumPy)__: A[i,j] = A[i*cols + j]
- __Column-major (Fortran, BLAS)__: A[i,j] = A[i + j*rows]

NumKong uses row-major to match Rust/Python/C conventions.

## Related Work

- __BLAS__: Basic Linear Algebra Subprograms (standard API)
- __cuBLAS__: NVIDIA's GPU-accelerated BLAS
- __Eigen__: C++ template library with block algorithms
- __OpenBLAS__: Open-source optimized BLAS
- __Intel MKL__: Proprietary, highly optimized for Intel CPUs
- __BLIS__: Portable BLAS-like library from AMD

## References

1. Goto, K., & Geijn, R. A. (2008). "Anatomy of high-performance matrix multiplication"
2. Low, T. M., et al. (2016). "Analytical Modeling Is Enough for High-Performance BLIS"
3. NVIDIA cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
