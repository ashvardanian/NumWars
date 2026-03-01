# Similarity Benchmarks

Vector similarity and distance metrics benchmarking module comparing NumKong against standard implementations.

## Overview

This module benchmarks __pairwise vector operations__ that take two vectors and output a scalar distance/similarity value. It covers three main categories:

1. __Spatial Similarity__: Geometric distances and angles
2. __Binary Similarity__: Set-based distances for binary vectors
3. __Probability Similarity__: Statistical divergences for probability distributions

## Supported Metrics

### Spatial Similarity

Metrics for continuous vector data:

- __Dot Product__: Inner product of two vectors
- __Angular Distance__: 1 - (cosine similarity between vectors)
- __Euclidean Distance__: √(Σ(aᵢ - bᵢ)²)
- __Squared Euclidean (sqeuclidean)__: Σ(aᵢ - bᵢ)²

__Supported types__: f64, f32, f16, bf16, i8

### Binary Similarity

Metrics for binary/set data:

- __Hamming Distance__: Count of differing bits
- __Jaccard Distance__: 1 - (intersection / union)

__Supported types__: u8, u4 (packed bits)

### Probability Similarity

Metrics for probability distributions:

- __Jensen-Shannon Divergence__: Symmetric KL divergence
- __Kullback-Leibler Divergence__: Relative entropy

__Supported types__: f64, f32, f16, bf16

## Usage

### Rust Benchmarks

```bash
# Run all similarity benchmarks
cargo bench --features bench_similarity --bench bench_similarity

# Run specific metric family
NUMWARS_METRIC=spatial cargo bench --features bench_similarity

# Run specific data types
NUMWARS_DTYPE=f32,f64 cargo bench --features bench_similarity

# Custom dimensions
NUMWARS_DIMS=512 cargo bench --features bench_similarity

# Filter by name pattern
NUMWARS_FILTER=angular cargo bench --features bench_similarity
```

### Python Benchmarks

```bash
# Run all similarity benchmarks
python similarity/bench.py

# Run specific metric family
python similarity/bench.py --metric dot

# Run specific data types
python similarity/bench.py --dtype f32,f64

# Custom dimensions and batch size
python similarity/bench.py --dimensions 512 --batch-size 10000

# All-pairs mode (cdist)
python similarity/bench.py --mode all-pairs
```

## Environment Variables

### Common

- `NUMWARS_DIMS`: Vector dimensions (default: 1536) (was: NUMWARS_DIMENSIONS)
- `NUMWARS_BATCH_SIZE`: Number of vector pairs (default: 1000)
- `NUMWARS_DTYPE`: Data type filter (default: "all")
- `NUMWARS_FILTER`: Benchmark name regex filter
- `NUMWARS_WARMUP_SECONDS`: Warmup time in seconds (default: 3.0) (was: NUMWARS_WARMUP)
- `NUMWARS_PROFILE_SECONDS`: Measurement time in seconds (default: 10.0) (was: NUMWARS_TIME_LIMIT)

### Similarity-Specific

- `NUMWARS_METRIC`: Metric family filter: "spatial", "binary", "probability" (default: "all")
- `NUMWARS_MODE`: "batch" or "all-pairs" (default: "batch", Python only)

## Benchmark Modes

### Batch Mode (Default)

Compute n pairwise distances: `[dist(a[i], b[i]) for i in range(n)]`

- Input: Two arrays of n vectors each
- Output: Array of n scalar distances
- Use case: Computing distances between corresponding pairs

### All-Pairs Mode (Python only)

Compute distance matrix: `[[dist(a[i], b[j]) for j in range(n)] for i in range(m)]`

- Input: Two arrays of m and n vectors
- Output: m×n distance matrix
- Use case: Finding nearest neighbors, clustering
- Equivalent to: `scipy.spatial.distance.cdist(A, B, metric)`

## Performance Metrics

All benchmarks report:

- __Throughput__: GB/s (data bandwidth)
- __Latency__: Time per operation
- __Operations/sec__: Number of distance calculations per second

For batch benchmarks with batch_size=1000 and dims=1536:
- Each operation processes 2 vectors × 1536 dims × dtype_size bytes
- f32: 2 × 1536 × 4 = 12,288 bytes per pair

## Example Results

Results will be added here after running benchmarks. Example format:

| Metric      | DType | NumKong (GB/s) | Baseline (GB/s) | Speedup |
| ----------- | ----- | -------------: | --------------: | ------: |
| dot         | f32   |           45.2 |            23.1 |   1.96× |
| angular     | f32   |           42.8 |            19.5 |   2.19× |
| sqeuclidean | f32   |           48.1 |            24.3 |   1.98× |
| hamming     | u8    |           38.7 |            15.2 |   2.55× |

## Competitors

### Rust

- __Baseline__: Manually unrolled SIMD-friendly implementations
- __ndarray__ (optional): Rust's de-facto array library

### Python

- __NumPy__: Vectorized operations with OpenBLAS/MKL backend
- __SciPy__: `spatial.distance` module (angular, sqeuclidean, hamming, jaccard, jensenshannon)
- __scikit-learn__: `metrics.pairwise` distances
- __PyTorch__ (optional): GPU-capable tensor operations
- __JAX__ (optional): JIT-compiled XLA backend
- __TensorFlow__ (optional): `tf.math` operations

## Implementation Notes

### Accuracy

All implementations are tested for numerical accuracy:
- Spatial metrics: ±1e-5 relative error for f32, ±1e-12 for f64
- Binary metrics: Exact integer matching
- Probability metrics: ±1e-4 relative error

### Optimization Techniques

NumKong uses:
- SIMD instructions (AVX-512, AVX2, NEON depending on CPU)
- Loop unrolling and vectorization
- Cache-friendly memory access patterns
- Type-specific optimizations (e.g., popcount for Hamming)

Baseline implementations use:
- 8-way manual loop unrolling
- Branch-free algorithms where possible
- Compile-time optimization hints

## Related Work

- __StringWars__: String operations benchmarking (sibling project)
- __SimSIMD__: Original library name (now NumKong)
- __BLAS__: Basic Linear Algebra Subprograms (standard for dot products)
- __scipy.spatial.distance__: Python's standard distance library
