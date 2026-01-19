# NumWars

__Mixed-precision numerical operations benchmarks: NumKong vs. the competition__

A comprehensive benchmarking suite comparing [NumKong](https://github.com/ashvardanian/SimSIMD) (formerly SimSIMD) against standard implementations for numerical operations across Rust and Python. Tests cover similarity metrics, elementwise operations, and matrix multiplication with support for exotic ML data types.

## Overview

NumWars provides production-grade benchmarks for three categories of numerical operations:

1. __[similarity/](similarity/)__ - Pairwise vector similarity/distance metrics
2. __[each/](each/)__ - Elementwise tensor operations
3. __[dots/](dots/)__ - Matrix multiplication (GEMM)

Each module includes both Rust and Python implementations, detailed documentation, and environment-variable-based configuration.

## Features

- __Comprehensive Coverage__: 3 benchmark modules, 15+ operations, 10+ data types
- __Both Languages__: Rust (Criterion) and Python implementations
- __Exotic Types__: Support for fp16, bf16, int8, int4, float8 (e4m3, e5m2)
- __Flexible Configuration__: Environment variables for fine-grained control
- __Multiple Competitors__: Compare against NumPy, SciPy, PyTorch, ndarray, BLAS, and more
- __Professional Infrastructure__: Modular design following StringWars patterns

## Quick Start

### Installation

#### Rust

```bash
# Clone the repository
git clone https://github.com/ashvardanian/NumWars.git
cd NumWars

# Run similarity benchmarks
cargo bench --features bench_similarity --bench bench_similarity

# Run all benchmarks
cargo bench --all-features
```

#### Python

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install specific modules
pip install -e ".[similarity]"  # Just similarity benchmarks
pip install -e ".[all]"         # Everything including PyTorch, JAX, TF

# Run similarity benchmarks
python similarity/bench.py

# Run elementwise benchmarks
python each/bench.py

# Run GEMM benchmarks
python dots/bench.py
```

## Benchmark Modules

### 1. Similarity (`similarity/`)

Pairwise vector distance and similarity metrics.

__Metrics:__
- __Spatial__: dot, cosine, L2, L2²
- __Binary__: hamming, jaccard
- __Probability__: jensen-shannon, kullback-leibler

__Data types__: f64, f32, f16, bf16, e4m3, e5m2, i8, i4, u8, u4

__Quick start:__
```bash
# Rust - run all benchmarks
cargo bench --features bench_similarity

# Rust - filter to specific benchmarks
NUMWARS_FILTER="f32" cargo bench --features bench_similarity

# Python - run all benchmarks
python similarity/bench.py

# Python - filter to specific metrics/types
NUMWARS_FILTER="cosine.*f32" python similarity/bench.py
```

See [similarity/README.md](similarity/README.md) for details.

### 2. Elementwise Operations (`each/`)

Component-wise tensor operations.

__Operations:__
- __Basic__: add, subtract, multiply, divide
- __Scaling__: scalar multiply/divide
- __Combined__: weighted sum (αA + βB), fused multiply-add

__Data types__: f64, f32, f16, bf16, e4m3, e5m2, i8-i64, u8-u64, i4, u4

__Quick start:__
```bash
# Rust - run all benchmarks
cargo bench --features bench_each

# Rust - configure tensor size and filter
NUMWARS_DIMS=2000000 NUMWARS_FILTER="add" cargo bench --features bench_each

# Python
python each/bench.py
```

See [each/README.md](each/README.md) for details.

### 3. Matrix Multiplication (`dots/`)

GEMM (General Matrix Multiply) operations computing C = A @ B.T.

__Layouts__: NT (No-Transpose × Transpose, NVIDIA convention)

__Data types__: f64, f32, f16, bf16, e4m3, e5m2, i8, i4

__Matrix dimensions__: Configurable m×n×k via environment variables

__Quick start:__
```bash
# Rust - run all benchmarks
cargo bench --features bench_dots

# Rust - configure rectangular matrices
NUMWARS_DIMS_WIDTH=2048 NUMWARS_DIMS_HEIGHT=1024 NUMWARS_DIMS_DEPTH=512 \
cargo bench --features bench_dots

# Python
python dots/bench.py
```

See [dots/README.md](dots/README.md) for details.

## Configuration

All benchmarks support configuration via environment variables. The key principle: **benchmarks generate all combinations of dtypes × operations by default**. Use `NUMWARS_FILTER` to selectively run specific benchmarks.

### Common Variables

```bash
# Universal filter - applies regex to full benchmark names
# Examples: "f32", "cosine|dot", "similarity/cosine/f32"
export NUMWARS_FILTER="f32"

# Timing parameters
export NUMWARS_WARMUP_SECONDS=3.0    # Warmup time in seconds (was: NUMWARS_WARMUP)
export NUMWARS_PROFILE_SECONDS=10.0  # Measurement time in seconds (was: NUMWARS_TIME_LIMIT)

# Rust-specific (Criterion)
export NUMWARS_SAMPLE_SIZE=50    # Number of samples
```

### Module-Specific Variables

#### Similarity

__Configuration__ (problem size, not filters):
```bash
export NUMWARS_DIMS=1536           # Vector dimensions (was: NUMWARS_DIMENSIONS)
export NUMWARS_BATCH_SIZE=1000     # Number of vector pairs
export NUMWARS_MODE=batch          # "batch" or "all-pairs"
```

__Filtering__ (use hierarchical names):
```bash
# Benchmark names: similarity/{library}/{metric}/{dtype}
# Examples: similarity/numkong/cosine/f32, similarity/baseline/dot/f64

NUMWARS_FILTER="f32"                      # Only f32 benchmarks
NUMWARS_FILTER="cosine|dot"               # Only cosine and dot metrics
NUMWARS_FILTER="similarity/numkong"       # Only NumKong library
```

#### Elementwise

__Configuration__:
```bash
export NUMWARS_DIMS=1000000        # Tensor size in elements (was: NUMWARS_SHAPE)
```

__Filtering__:
```bash
# Benchmark names: each/{library}/{operation}/{dtype}
# Examples: each/baseline/add/f32, each/numkong/multiply/f64

NUMWARS_FILTER="add"               # Only add operations
NUMWARS_FILTER="f32"               # Only f32 benchmarks
```

#### Matrix Multiplication

__Configuration__ (supports rectangular matrices m×n×k):
```bash
export NUMWARS_DIMS_WIDTH=1024     # Matrix C width (n) (was: NUMWARS_WIDTH)
export NUMWARS_DIMS_HEIGHT=1024    # Matrix C height (m) (was: NUMWARS_HEIGHT)
export NUMWARS_DIMS_DEPTH=1024     # Shared dimension (k) (was: NUMWARS_DEPTH)
export NUMWARS_THREADS=8           # Thread count for parallel benchmarks
```

__Filtering__:
```bash
# Benchmark names: dots/{library}/{dtype}/{m}x{n}x{k}/{threads}t
# Examples: dots/numkong/f32/1024x1024x1024/8t, dots/ndarray/f32/1024x1024x1024/1t

NUMWARS_FILTER="f32"                # Only f32 benchmarks
NUMWARS_FILTER="1024x1024"          # Specific matrix sizes
NUMWARS_FILTER="dots/ndarray"       # Only ndarray library
```

## Project Structure

```
NumWars/
├── Cargo.toml              # Rust workspace manifest
├── pyproject.toml          # Python project configuration
├── requirements.txt        # Python dependencies
├── utils.rs                # Shared Rust utilities
├── utils.py                # Shared Python utilities
├── README.md               # This file
├── LICENSE                 # Apache 2.0
│
├── similarity/             # Similarity benchmarks
│   ├── bench.rs           # Rust benchmarks
│   ├── bench.py           # Python benchmarks
│   └── README.md          # Module documentation
│
├── each/                   # Elementwise operations
│   ├── bench.rs           # Rust benchmarks
│   ├── bench.py           # Python benchmarks
│   └── README.md          # Module documentation
│
└── dots/                   # Matrix multiplication
    ├── bench.rs           # Rust benchmarks
    ├── bench.py           # Python benchmarks
    └── README.md          # Module documentation
```

## Data Types

NumWars supports a wide range of numerical types:

### Standard Types

| Type | Size | Range | Use Case |
|------|-----:|------:|----------|
| f64 | 8 bytes | ±1.7e308 | Scientific computing |
| f32 | 4 bytes | ±3.4e38 | General purpose |
| i64/u64 | 8 bytes | ±9.2e18 / 0-1.8e19 | Large integers |
| i32/u32 | 4 bytes | ±2.1e9 / 0-4.3e9 | Standard integers |
| i16/u16 | 2 bytes | ±32K / 0-65K | Compact integers |
| i8/u8 | 1 byte | ±128 / 0-255 | Quantized values |

### ML-Specific Types (via ml_dtypes)

| Type | Size | Exponent | Mantissa | Use Case |
|------|-----:|---------:|---------:|----------|
| f16 | 2 bytes | 5 bits | 10 bits | Half precision |
| bf16 | 2 bytes | 8 bits | 7 bits | Brain Float (Google TPU) |
| e4m3 | 1 byte | 4 bits | 3 bits | FP8 training |
| e5m2 | 1 byte | 5 bits | 2 bits | FP8 inference |
| i4 | 0.5 bytes | - | - | Extreme quantization |
| u4 | 0.5 bytes | - | - | Packed binary data |

__Python support__: Install `ml_dtypes` for exotic types:
```bash
pip install ml_dtypes
```

## Competitors

### Rust

- __Baseline__: Hand-optimized reference implementations
- __NumKong__: Primary library being benchmarked (SimSIMD)
- __ndarray__: De-facto standard array library
- __nalgebra__: Linear algebra library
- __faer__: Modern, fast linear algebra (f32/f64 only for GEMM)

### Python

- __NumPy__: Universal standard (OpenBLAS/MKL backend)
- __SciPy__: Scientific computing library
- __scikit-learn__: Machine learning utilities
- __PyTorch__: Deep learning framework (optional)
- __JAX__: JIT-compiled arrays (optional)
- __TensorFlow__: ML platform (optional)

## Performance Metrics

All benchmarks report:

- __Throughput (GB/s)__: Memory bandwidth utilization
- __Operations/sec__: Auto-scaling format (KiloOps/s, MegaOps/s, GigaOps/s, TeraOps/s)
  - Similar to StringWars CUPS (Characters Used Per Second) formatter
  - Automatically selects appropriate scale based on magnitude
- __Latency__: Time per operation
- __Speedup__: Relative to baseline/competitor

Example output:
```
similarity/numkong/cosine/f32       2.15 µs   45.2 GB/s   234 MegaOps/s
similarity/numkong/dot/f64          3.82 µs   51.3 GB/s   187 MegaOps/s
each/numkong/add/f32                1.23 µs   42.1 GB/s   1.23 GigaOps/s
dots/numkong/f32/1024x1024x1024/8t  8.45 ms   215 GigaOps/s
```

## Example Usage

### Running Specific Benchmarks

```bash
# Run only f32 cosine similarity
NUMWARS_FILTER="cosine.*f32" cargo bench --features bench_similarity

# Run 512-dimensional vectors
NUMWARS_DIMS=512 python similarity/bench.py

# Run only addition with 1M elements
NUMWARS_DIMS=1000000 cargo bench --features bench_each

# Run GEMM with 2048×2048 matrices
NUMWARS_DIMS_WIDTH=2048 NUMWARS_DIMS_HEIGHT=2048 NUMWARS_DIMS_DEPTH=2048 \
python dots/bench.py

# Multi-threaded GEMM (8 threads)
NUMWARS_THREADS=8 cargo bench --features bench_dots
```

### Comparing Libraries

```bash
# Python: Compare NumKong vs NumPy vs SciPy
python similarity/bench.py --dtype f32 --metric spatial

# Rust: Compare NumKong vs baseline vs ndarray
cargo bench --features bench_each --bench bench_each
```

## Development

### Building from Source

```bash
# Clone with submodules (for NumKong/SimSIMD)
git clone --recursive https://github.com/ashvardanian/NumWars.git
cd NumWars

# Build Rust benchmarks
cargo build --release --all-features

# Install Python package in development mode
pip install -e ".[all]"
```

### Adding New Benchmarks

1. Create benchmark function in `module/bench.rs` or `module/bench.py`
2. Use `should_run_benchmark()` for filtering
3. Report throughput metrics
4. Add documentation to module README

See existing benchmarks for examples.

### Testing

```bash
# Rust: Run tests
cargo test --all-features

# Python: Run with small dataset
NUMWARS_DIMS=128 NUMWARS_PROFILE_SECONDS=1.0 python similarity/bench.py
```

## Results

Benchmark results will be published here after running on reference hardware.

### Test Platform

- __CPU__: (To be determined)
- __RAM__: (To be determined)
- __OS__: Ubuntu 22.04 LTS
- __Compiler__: Rust 1.75.0, GCC 11.4.0
- __BLAS__: OpenBLAS 0.3.21

## Related Projects

- __[StringWars](https://github.com/ashvardanian/StringWars)__: String operations benchmarks (sibling project)
- __[NumKong/SimSIMD](https://github.com/ashvardanian/SimSIMD)__: The library being benchmarked
- __[USearch](https://github.com/unum-cloud/usearch)__: Vector search using NumKong
- __[ml_dtypes](https://github.com/jax-ml/ml_dtypes)__: Google's ML data types library

## Contributing

Contributions welcome! Please:

1. Follow the existing code style
2. Add tests for new benchmarks
3. Update documentation
4. Run benchmarks before submitting PR

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Citation

If you use NumWars in your research, please cite:

```bibtex
@software{numwars2024,
  title = {NumWars: Mixed-Precision Numerical Operations Benchmarks},
  author = {NumWars Contributors},
  year = {2024},
  url = {https://github.com/ashvardanian/NumWars}
}
```

## Acknowledgments

- Inspired by [StringWars](https://github.com/ashvardanian/StringWars)
- Built on [NumKong/SimSIMD](https://github.com/ashvardanian/SimSIMD)
- Uses [Criterion](https://github.com/bheisler/criterion.rs) for Rust benchmarks
- Uses [ml_dtypes](https://github.com/jax-ml/ml_dtypes) for exotic types
