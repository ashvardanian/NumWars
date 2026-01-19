# Elementwise Operations Benchmarks

Elementwise (component-wise) tensor operations benchmarking module comparing NumKong Tensor API against standard implementations.

## Overview

This module benchmarks __elementwise operations__ that map vectors and scalars to output vectors. These are fundamental building blocks for neural networks, scientific computing, and array processing.

## Supported Operations

### Basic Arithmetic

- __Addition__: `R = A + B` (vector + vector)
- __Subtraction__: `R = A - B` (vector - vector)
- __Multiplication__: `R = A * B` (vector * vector, element-wise)
- __Division__: `R = A / B` (vector / vector, element-wise)

### Scaling

- __Scalar Multiplication__: `R = alpha * A` (scalar * vector)
- __Scalar Division__: `R = A / alpha` (vector / scalar)

### Combined Operations

- __Weighted Sum__: `R = alpha * A + beta * B` (AXPY-like)
- __Fused Multiply-Add__: `R = alpha * A * B + beta * C`

### In-Place Operations

- __Add In-Place__: `A += B`
- __Scale In-Place__: `A *= alpha`
- __Multiply In-Place__: `A *= B`

## Supported Data Types

### Floating Point

- __f64__: 64-bit IEEE 754 double precision
- __f32__: 32-bit IEEE 754 single precision
- __f16__: 16-bit IEEE 754 half precision
- __bf16__: 16-bit Brain Float (Google/Intel)
- __e4m3__: 8-bit float with 4-bit exponent, 3-bit mantissa
- __e5m2__: 8-bit float with 5-bit exponent, 2-bit mantissa

### Integer

- __i64, i32, i16, i8__: Signed integers
- __u64, u32, u16, u8__: Unsigned integers
- __i4, u4__: 4-bit integers (packed)

## Usage

### Rust Benchmarks

```bash
# Run all elementwise benchmarks
cargo bench --features bench_each --bench bench_each

# Run specific operation
NUMWARS_OPERATION=add cargo bench --features bench_each

# Run specific data types
NUMWARS_DTYPE=f32,f64 cargo bench --features bench_each

# Custom tensor size
NUMWARS_DIMS=1000000 cargo bench --features bench_each

# Filter by name pattern
NUMWARS_FILTER=add cargo bench --features bench_each
```

### Python Benchmarks

```bash
# Run all elementwise benchmarks
python each/bench.py

# Run specific operation
python each/bench.py --operation add

# Run specific data types
python each/bench.py --dtype f32,f64

# Custom tensor size
python each/bench.py --shape 5000000

# Test broadcasting
python each/bench.py --broadcast
```

## Environment Variables

### Common

- `NUMWARS_DTYPE`: Data type filter (default: "f32")
- `NUMWARS_FILTER`: Benchmark name regex filter
- `NUMWARS_WARMUP_SECONDS`: Warmup time in seconds (default: 3.0) (was: NUMWARS_WARMUP)
- `NUMWARS_PROFILE_SECONDS`: Measurement time in seconds (default: 10.0) (was: NUMWARS_TIME_LIMIT)

### Elementwise-Specific

- `NUMWARS_DIMS`: Tensor size in elements (default: 1000000) (was: NUMWARS_SHAPE)
- `NUMWARS_OPERATION`: Operation filter: "add", "multiply", "scale", "wsum", "fma" (default: "all")
- `NUMWARS_BROADCAST`: Test broadcasting (default: true, Python only)

## Performance Metrics

All benchmarks report:

- __Throughput__: GB/s (memory bandwidth utilization)
- __Elements/sec__: Number of elements processed per second
- __Latency__: Time per operation

For a 1M element f32 array:
- Memory: 4 MB per array
- Add: reads 8 MB, writes 4 MB = 12 MB total
- Throughput = 12 MB / duration

## Example Results

Results will be added here after running benchmarks. Example format:

| Operation | DType | NumKong (GB/s) | NumPy (GB/s) | ndarray (GB/s) | Speedup vs NumPy |
| --------- | ----- | -------------: | -----------: | -------------: | ---------------: |
| add       | f32   | 42.5           | 38.2         | 35.1           | 1.11×            |
| multiply  | f32   | 44.1           | 39.5         | 36.8           | 1.12×            |
| scale     | f32   | 51.2           | 45.3         | 42.1           | 1.13×            |
| wsum      | f32   | 38.9           | 32.1         | 29.5           | 1.21×            |
| fma       | f32   | 35.7           | 28.4         | 26.2           | 1.26×            |

## Broadcasting

Python benchmarks support NumPy-style broadcasting:

- __Scalar-Vector__: `scalar + vector`
- __Vector-Vector__: `vector1 + vector2` (same shape)
- __Row-Matrix__: `row_vector + matrix` (broadcast across rows)
- __Column-Matrix__: `col_vector + matrix` (broadcast across columns)

Broadcasting follows NumPy semantics and is tested for correctness.

## Competitors

### Rust

- __Baseline__: Iterative and unrolled implementations
- __ndarray__: Rust's de-facto array library with optimized elementwise ops
- __nalgebra__: Linear algebra library (component-wise operations)

### Python

- __NumPy__: Universal functions (ufuncs) with OpenBLAS/MKL backend
- __PyTorch__ (optional): GPU-capable tensor operations
- __JAX__ (optional): JIT-compiled XLA backend
- __TensorFlow__ (optional): `tf.math` operations

## Implementation Notes

### Memory Access Patterns

Elementwise operations are __memory-bound__, not compute-bound. Performance depends on:

1. __Cache efficiency__: Sequential access patterns
2. __Prefetching__: Hardware and software prefetch
3. __SIMD width__: Processing multiple elements per instruction
4. __Memory bandwidth__: RAM speed limits throughput

Theoretical peak throughput for DDR4-3200:
- Bandwidth: ~25 GB/s per channel
- Dual-channel: ~50 GB/s
- Elementwise add (read 2, write 1): ~16 GB/s expected

### In-Place vs Out-of-Place

__In-place operations__ (`A += B`) are faster because:
- Fewer memory allocations
- Less data movement (2 reads + 1 write → 1 read + 1 write)
- Better cache locality

__Out-of-place operations__ (`C = A + B`) are safer:
- No side effects
- Original data preserved
- Easier to parallelize

### Vectorization

NumKong uses SIMD intrinsics for vectorization:
- AVX-512: 512-bit registers (16× f32, 8× f64)
- AVX2: 256-bit registers (8× f32, 4× f64)
- NEON: 128-bit registers (4× f32, 2× f64)

Baseline implementations rely on compiler auto-vectorization.

## Related Work

- __NumPy ufuncs__: Universal functions for elementwise operations
- __Eigen__: C++ template library for linear algebra
- __ArrayFire__: GPU library for array operations
- __cuBLAS__: NVIDIA's GPU-accelerated BLAS implementation
