# NumWars

## Mixed-Precision Numerics Benchmarks for Rust & Python

![NumWars banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/NumWars-v1.png?raw=true)

There are many strong libraries for numerical computing.
Most of them are written in C, C++, and Fortran, with excellent Rust wrappers and Python bindings on top.

Where Rust is especially convenient is dependency management and reproducible benchmarking, making it a good place to line up apples-to-apples comparisons across native crates and their Python bindings.
NumWars exists for the same reason StringWars exists for StringZilla: to compare [NumKong](https://github.com/ashvardanian/NumKong) against mainstream CPU and GPU stacks on the workloads it was built for, including:

- [`ndarray`](https://github.com/rust-ndarray/ndarray) and [`nalgebra`](https://github.com/dimforge/nalgebra) for dense tensor and linear algebra kernels.
- [`faer`](https://github.com/sarah-quinones/faer-rs) and [`matrixmultiply`](https://github.com/bluss/matrixmultiply) for GEMM-like Rust baselines.
- [`geo`](https://github.com/georust/geo) for geographic distances.
- [`polars`](https://github.com/pola-rs/polars) and reduction-heavy analytics workloads.
- [`NumPy`](https://github.com/numpy/numpy), [`SciPy`](https://github.com/scipy/scipy), and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) on Python.
- cuBLAS and cuBLASLt through [`cudarc`](https://github.com/chelsea0x3b/cudarc) in Rust, and [`PyTorch`](https://github.com/pytorch/pytorch) and [`CuPy`](https://github.com/cupy/cupy) in Python, for the GPU rows.

Of course, the APIs and internal kernels of those projects are different.
So this repository focuses on the workload families NumKong was designed for and compares their effective throughput using the native unit for each operation family instead of forcing everything into fake global `ops/s`.

> [!IMPORTANT]
> The charts below were measured single-threaded on one core of an Intel Xeon5 Emerald Rapids, on October 14, 2026.
> Each suite README keeps its reference numbers per machine, in an Intel Xeon5 Emerald Rapids section and an Apple M5 Pro section, both single-threaded.
> The `dots`, `similarities` and `attention` READMEs add a 148× Nvidia SM100 section with GPU rows.
> They will vary with CPU model, compiler flags, BLAS backend, and problem size.
> Rebuild and rerun on your own hardware before treating them as absolute.

## Benchmarks at a Glance

### Packed Matrix Multiplication

NumKong packed dots are mixed-precision by design.
_i8_ inputs produce _i32_ outputs.
_bf16_ and _f16_ inputs produce _f32_ outputs.
_f32_ inputs produce _f64_ outputs.
The mainstream baselines shown here keep _f32 → f32_.
Every product below is C = A × Bᵀ with A of 1024×1536 and B of 128×1536.
Compared to Rust projects, it means:

```text
NumKong:
numkong::Tensor::dots_packed i8 → i32   ███████████████████████████████████ 2,876.00 GSO/s
numkong::Tensor::dots_packed bf16 → f32 ████████████████▍                   1,346.70 GSO/s
numkong::Tensor::dots_packed f16 → f32  █▋                                    137.44 GSO/s
numkong::Tensor::dots_packed f32 → f64  ▋                                      53.77 GSO/s

Alternatives:
ndarray::ArrayBase::dot f32 → f32       ██▏                                   182.44 GSO/s
faer::linalg::matmul::matmul f32 → f32  █▊                                    150.96 GSO/s
matrixmultiply::sgemm f32 → f32         █                                      91.29 GSO/s
nalgebra::DMatrix × DMatrixᵀ f32 → f32  █                                      87.32 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.dots_packed i8 → i32   ████████████████████████████████████████████ 1,481.06 GSO/s
numkong.dots_packed bf16 → f32 ████████████████████▌                          691.38 GSO/s
numkong.dots_packed f16 → f32  █▊                                              61.21 GSO/s
numkong.dots_packed f32 → f64  ▉                                               31.46 GSO/s

Alternatives:
numpy.matmul f32 → f32         █████▌                                         187.66 GSO/s
```

See [dots/README.md](dots/README.md) for details.

### Pairwise Similarity

Single-pair vector kernels at 1536 dimensions.
This lists _Dot_ products and true _Euclidean_ distances measurements into one throughput-sorted view.
NumKong keeps its mixed-precision promotions, while the baseline libraries mostly stay in their input type.

Compared to Rust projects, it means:

```text
NumKong:
numkong::Dot::dot u8 → u32               █████████████████████████████████████ 65.91 GSO/s
numkong::Dot::dot i8 → i32               ████████████████████████████████▉     58.63 GSO/s
numkong::Euclidean::euclidean u8 → f32   █████████████████████████             44.67 GSO/s
numkong::Euclidean::euclidean i8 → f32   ██████████████████████▌               40.11 GSO/s
numkong::Dot::dot bf16 → f32             ████████████▉                         23.05 GSO/s
numkong::Euclidean::euclidean bf16 → f32 ████████▍                             14.98 GSO/s
numkong::Dot::dot f32 → f64              ████▍                                  7.80 GSO/s
numkong::Euclidean::euclidean f32 → f64  ███▍                                   6.12 GSO/s

Alternatives:
ndarray::ArrayBase::dot f32 → f32        ███████████▋                          20.72 GSO/s
nalgebra::Matrix::dot f32 → f32          █████████                             16.20 GSO/s
nalgebra (a - b).norm() f32 → f32        █████▍                                 9.72 GSO/s
ndarray sqrt((a - b)·(a - b)) f32 → f32  █████▍                                 9.65 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.euclidean u8 → f32  ███████████████████████████████████████████████████ 6.13 GSO/s
numkong.euclidean i8 → f32  █████████████████████████████████████████████████   5.90 GSO/s
numkong.dot u8 → f32        ████████████████████████████████████████████████▉   5.88 GSO/s
numkong.angular u8 → f32    ████████████████████████████████████████████████▊   5.86 GSO/s
numkong.dot i8 → f32        █████████████████████████████████████████████▌      5.48 GSO/s
numkong.angular i8 → f32    ███████████████████████████████████████████▎        5.21 GSO/s
numkong.euclidean f32 → f32 ██████████████████████████████▏                     3.63 GSO/s
numkong.dot f64 → f32       ███████████████████████▎                            2.80 GSO/s

Alternatives:
numpy.dot f32 → f32         ██████████████▋                                     1.76 GSO/s
numpy.dot f64 → f64         ████████████▉                                       1.55 GSO/s
```

See [similarity/README.md](similarity/README.md) for details.

### All-Pairs Similarity Matrices

Matrix-vs-matrix comparisons at 2048 rows by 1536 dimensions.
These are the packed many-to-many siblings of the pairwise spatial kernels above.
The merged lists below include _angular_ and _euclidean_ metrics, and the headline unit is GSO/s.

Compared to Rust projects, it means:

```text
NumKong:
numkong::Tensor::euclideans_packed i8 → f32   ███████████████████████████████ 712.29 GSO/s
numkong::Tensor::euclideans_packed u8 → f32   ██████████████████████████████▊ 708.90 GSO/s
numkong::Tensor::angulars_packed u8 → f32     █████████████████████████████▏  670.61 GSO/s
numkong::Tensor::angulars_packed i8 → f32     █████████████████████████████   667.48 GSO/s
numkong::Tensor::euclideans_packed bf16 → f32 █████████████████▏              395.44 GSO/s
numkong::Tensor::angulars_packed bf16 → f32   ████████████████▉               388.92 GSO/s
numkong::Tensor::euclideans_packed f32 → f64  █                                25.13 GSO/s
numkong::Tensor::angulars_packed f32 → f64    █                                24.42 GSO/s

Alternatives:
ndarray euclidean matrix f32 → f32            ████▏                            96.88 GSO/s
ndarray angular matrix f32 → f32              ████▏                            95.16 GSO/s
nalgebra euclidean matrix f32 → f32           █▌                               35.65 GSO/s
nalgebra angular matrix f32 → f32             █▌                               35.17 GSO/s
```

Compared to Python through SciPy `cdist`:

```text
NumKong:
numkong.euclideans_packed u8 → f32  █████████████████████████████████████████ 549.79 GSO/s
numkong.euclideans_packed i8 → f32  ████████████████████████████████████████▌ 544.76 GSO/s
numkong.angulars_packed u8 → f32    ██████████████████████████████████████▉   522.92 GSO/s
numkong.angulars_packed i8 → f32    ██████████████████████████████████████▉   521.78 GSO/s
numkong.euclideans_packed f32 → f64 █▏                                         15.56 GSO/s
numkong.angulars_packed f32 → f64   █▏                                         15.41 GSO/s

Alternatives:
scipy.cdist euclidean f32 → f64     ▎                                           3.49 GSO/s
scipy.cdist cosine f32 → f64        ▏                                           2.57 GSO/s
```

See [similarities/README.md](similarities/README.md) for details.

### Elementwise Operations

Bandwidth-sensitive elementwise kernels over 2048 elements, covering sum, scale and fused multiply-add.
Sum shown as representative sample.
In Rust:

```text
NumKong:
numkong::EachSum f32 → f32 ███████████████████████████████████████████████████ 251.18 GB/s
numkong::EachSum i8 → i8   ████████████████████████████████████▎               178.63 GB/s
numkong::EachSum f16 → f16 ███████████████████████████████▏                    153.74 GB/s

Alternatives:
serial code f32 → f32      ██████████████████████████████████████              187.72 GB/s
nalgebra::add f32 → f32    ████████████████████████████████▎                   159.28 GB/s
ndarray::add f32 → f32     ███████████████████████████████▎                    154.08 GB/s
```

In Python:

```text
numpy.add f32 → f32     ███████████████████████████████████████████████████████ 45.90 GB/s
numkong.add f64 → f64   ██████████████████████████████████████████████████████▊ 45.73 GB/s
numpy.add f64 → f64     ███████████████████████████████████████████▌            36.40 GB/s
numkong.add f32 → f32   █████████████████████████████████████████▋              34.78 GB/s
numkong.add f16 → f16   ███████████████████████▊                                19.91 GB/s
numpy.add i8 → i8       █████████████████▊                                      14.86 GB/s
numkong.add i8 → i8     ████████████▎                                           10.28 GB/s
numkong.add bf16 → bf16 ███                                                      2.54 GB/s
numpy.add f16 → f16     █▏                                                       1.01 GB/s
```

See [each/README.md](each/README.md) for details.

### Reductions

Horizontal reductions over 2048 elements.
The suite covers sum, min-max and row-wise L2 norms.
In Rust:

```text
ndarray::ArrayBase::sum f64 → f64       ██████████████████████████████████████ 103.34 GB/s
polars::ChunkedArray::sum f64 → f64     █████████████████████████████████████▎ 101.31 GB/s
ndarray::ArrayBase::sum f32 → f32       ███████████████████████▊                64.74 GB/s
numkong::reduce_moments().sum f32 → f64 ██████████████████▋                     50.73 GB/s
polars::ChunkedArray::sum f32 → f32     ██████▋                                 18.13 GB/s
serial sum loop f32 → f32               ██▊                                      7.64 GB/s
```

Row-wise L2 norms over a 2048×1536 matrix:

```text
ndarray row norms f64 → f64             ███████████████████████████████████████ 26.16 GB/s
ndarray row norms f32 → f32             █████████████████████████████████████   24.83 GB/s
numkong::Dot self-dot + sqrt bf16 → f32 ████████████████████████████████████▋   24.62 GB/s
numkong::Dot self-dot + sqrt f64 → f64  █████████████████████████████▎          19.64 GB/s
numkong::Dot self-dot + sqrt f32 → f64  ██████████████████████████▉             18.08 GB/s
serial row norms loop f64 → f64         █████████████████████▎                  14.31 GB/s
serial row norms loop f32 → f32         ███████████▎                             7.59 GB/s
```

In Python over 2048 elements, and over the same 2048×1536 matrix for the norms:

```text
numkong.sum f32 → f32       ███████████████████████████████████████████████████ 32.05 GB/s
numkong.sum f64 → f64       ██████████████████████████████████████              23.91 GB/s
numkong.norm f32 → f64      █████████████████████████████████████▋              23.70 GB/s
numkong.norm f64 → f64      ██████████████████████████████▊                     19.33 GB/s
numkong.sum u8 → u8         ███████████████████▎                                12.14 GB/s
numkong.sum i8 → i8         ██████████████████▉                                 11.93 GB/s
numpy.sum f64 → f64         ████████████████▏                                   10.16 GB/s
numpy.linalg.norm f32 → f64 ███████████▌                                         7.23 GB/s
numpy.linalg.norm f64 → f64 ███████████▎                                         7.12 GB/s
numpy.sum f32 → f32         ████████▏                                            5.12 GB/s
numpy.sum i8 → i8           █▎                                                   0.85 GB/s
```

See [reduce/README.md](reduce/README.md) for details.

### MaxSim

ColBERT-style late interaction with 1024 query vectors, 128 document vectors, and 1536 dimensions.
NumKong promotes _f32 → f64_ here as well, while ndarray stays in _f32_.
In Rust:

```text
NumKong:
numkong::MaxSimPackedMatrix::score bf16 → f32 █████████████████████████████ 1,320.62 GSO/s
numkong::MaxSimPackedMatrix::score f16 → f32  ▋                                30.85 GSO/s
numkong::MaxSimPackedMatrix::score f32 → f64  ▍                                15.01 GSO/s

Alternatives:
ndarray Q × Dᵀ max-reduce f32 → f32           ██                               91.82 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.maxsim_packed bf16 → f32 ██████████████████████████████████████████ 1,237.41 GSO/s
numkong.maxsim_packed f16 → f32  ██████▌                                      193.64 GSO/s
numkong.maxsim_packed f32 → f64  ███▋                                         109.82 GSO/s

Alternatives:
numpy matmul f32 → f32           █████▉                                       175.15 GSO/s
```

See [maxsim/README.md](maxsim/README.md) for details.

### Geospatial Distances

Many-to-many distances from 1024 row points to 128 column points, each set packed once outside the timed loop.
The unit is MP/s, or million coordinate pairs per second.
The merged lists below include _Haversine_ distances on the IUGG mean sphere and _Vincenty_ distances on the WGS-84 ellipsoid, where `geo` and pyproj answer with Karney's geodesic solver.

Compared to Rust projects, it means:

```text
NumKong:
numkong::HaversinesPackedMatrix f64 → f64 ████████████████████████████████████ 303.91 MP/s
numkong::VincentysPackedMatrix f64 → f64  ██▊                                   24.18 MP/s

Alternatives:
geo::Haversine distance f64 → f64         ██▊                                   23.78 MP/s
geo::Geodesic distance f64 → f64          ▏                                      1.39 MP/s
```

Compared to Python and its alternatives:

```text
NumKong:
numkong.haversines_packed f64 → f64   ████████████████████████████████████████ 292.20 MP/s
numkong.vincentys_packed f64 → f64    ███▎                                      24.40 MP/s

Alternatives:
NumPy broadcast haversine f64 → f64   ████▋                                     34.05 MP/s
sklearn haversine_distances f64 → f64 ██▍                                       17.59 MP/s
pyproj.Geod.inv f64 → f64             ▏                                          1.24 MP/s
```

See [geospatial/README.md](geospatial/README.md) for details.

### Shape Distances

Many-to-many distances between 8 row and 8 column clouds of 32 3D points each, with the correspondence-free _Chamfer_, _Hausdorff_, _Wasserstein-Procrustes_ and _Gromov-Wasserstein_ distances that replaced mesh alignment.
Every distance takes the best of the 8 sign flips of each cloud's principal axes, and the baselines search the same flips.
The unit is MP/s, or million cloud pairs per second, and the lists below cover _f32_ clouds.
In Rust:

```text
NumKong:
numkong::ShapesPackedMatrix::chamfers f32 → f64              █████████████████ 0.6271 MP/s
numkong::ShapesPackedMatrix::hausdorffs f32 → f64            ████████████████▉ 0.6265 MP/s
numkong::ShapesPackedMatrix::wassersteinprocrustes f32 → f64                   0.0023 MP/s
numkong::ShapesPackedMatrix::gromovwassersteins f32 → f64                      0.0009 MP/s

Alternatives:
nalgebra-based Chamfer f32 → f64                             ██▎               0.0858 MP/s
nalgebra-based Hausdorff f32 → f64                           ██▎               0.0858 MP/s
lapjv-based Wasserstein-Procrustes f32 → f64                                   0.0028 MP/s
```

Compared to Python and its alternatives:

```text
NumKong:
numkong.chamfers_packed f32 → f64              ███████████████████████████████ 0.6214 MP/s
numkong.hausdorffs_packed f32 → f64            ██████████████████████████████▍ 0.6095 MP/s
numkong.wassersteinprocrustes_packed f32 → f64                                 0.0023 MP/s
numkong.gromovwassersteins_packed f32 → f64                                    0.0009 MP/s

Alternatives:
scipy.spatial.cKDTree Hausdorff f32 → f64      ▏                               0.0039 MP/s
scipy.spatial.cKDTree Chamfer f32 → f64        ▏                               0.0036 MP/s
ot.gromov.gromov_wasserstein2 f32 → f64                                        0.0014 MP/s
SciPy Wasserstein-Procrustes f32 → f64                                         0.0005 MP/s
```

See [mesh/README.md](mesh/README.md) for details.

## Replicating the Results

### Rust

Every Rust benchmark is a Criterion harness behind a Cargo feature gate.
Run one suite at a time or all at once:

```bash
# One suite
RUSTFLAGS="-C target-cpu=native" \
cargo bench --features bench_similarity --bench bench_similarity

# All suites
RUSTFLAGS="-C target-cpu=native" \
cargo bench --features all
```

Tuning knobs (environment variables):

| Variable                 | Default | Meaning                                                   |
| :----------------------- | :------ | :-------------------------------------------------------- |
| `NUMWARS_FILTER`         |         | Regex over benchmark names, or a substring if not a regex |
| `NUMWARS_SEED`           | 42      | 32-bit seed for random inputs, or `random`                |
| `NUMWARS_WARMUP`         | 1s      | Warm-up per benchmark, like `200ms` or `1s`               |
| `NUMWARS_TIME_LIMIT`     | 10s     | Measurement time per benchmark, like `200ms` or `10s`     |
| `NUMWARS_BATCH_PER_CORE` | 2048    | Items per call, times the threads: elements, rows, points |
| `NUMWARS_THREADS`        | 1       | Threads for NumKong and competitors, `0` for all cores    |
| `NUMWARS_DIMS`           | 1536    | Vector length                                             |
| `NUMWARS_DIMS_HEIGHT`    | 1024    | Rows of `C = A × Bᵀ` in GEMM-shaped suites                |
| `NUMWARS_DIMS_WIDTH`     | 128     | Columns of `C = A × Bᵀ` in GEMM-shaped suites             |
| `NUMWARS_DIMS_DEPTH`     | 1536    | Shared dimension of `A` and `B` in GEMM-shaped suites     |
| `NUMWARS_DEVICES`        | cuda:0  | CUDA devices in GPU suites, like `cuda:0,cuda:1`          |
| `NUMWARS_SAMPLE_SIZE`    | 50      | Criterion samples per benchmark, Rust only                |
| `NUMWARS_OUTPUT`         | table   | Human-readable `table` or machine-readable `json`, Python |
| `NUMWARS_SCIPY`          | false   | Whether the Python `similarity` suite also times SciPy    |

`NUMWARS_DEVICES` is read by the suites that put rows on a GPU, `dots`, `similarities` and `attention`, and any other suite refuses it rather than ignoring it.
Each selected device gets its own rows, named after its ordinal as in `<cuda:1>`, and a device named in the variable that cannot be opened stops the run.
Without the variable, a GPU build puts rows on device zero when the machine has one.
Every GPU row synchronizes inside its timed region, and checks its first round against NumKong's CPU kernel on the leading rows before timing.
The suite defaults are CPU-sized, so GPU runs usually raise the shapes and filter on `cuda`, which only GPU rows carry.

In Rust, the `cuda` feature adds those rows, building NumKong's CUDA kernels with `nvcc`, or linking the CMake tree `NUMKONG_LIBRARY_DIR` names, and loading cuBLAS and cuBLASLt from the CUDA 13 toolkit at run time:

```bash
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda \
NUMWARS_DIMS_HEIGHT=16384 NUMWARS_DIMS_WIDTH=16384 NUMWARS_DIMS_DEPTH=16384 \
cargo bench --features "bench_dots cuda" --bench bench_dots
```

NumKong's Rust verbs dispatch to the CPU, so these rows call its exported C dispatch with the device's capabilities and stream, over operands in its `UnifiedAllocator`.
In Python, they need NumKong built with `-C cmake.define.NUMKONG_BUILD_CUDA=ON`, PyTorch's CUDA wheels to hold the operands, and optionally CuPy:

```bash
pip install -e ".[cuda]" --extra-index-url https://download.pytorch.org/whl/cu130
pip install ../NumKong -C cmake.define.NUMKONG_BUILD_CUDA=ON
NUMWARS_DEVICES=cuda:0 NUMWARS_FILTER=cuda NUMWARS_BATCH_PER_CORE=16384 python similarities/bench.py
```

The Python suites read the same variables, and take no flags.
An empty variable counts as unset, and a value that does not parse stops the run.

### Python

Install with `uv` and run any suite directly:

```bash
uv run --with "numkong,numpy,scipy,tabulate,ml_dtypes" \
python similarity/bench.py
```

Or install all extras and run from the repo root:

```bash
pip install -e ".[similarity,each,dots,geospatial,mesh,reduce,similarities]"
python dots/bench.py
python similarities/bench.py
```

## Benchmark Suites

- [similarity/README.md](similarity/README.md)
- [similarities/README.md](similarities/README.md), with GPU rows
- [dots/README.md](dots/README.md), with GPU rows
- [each/README.md](each/README.md)
- [reduce/README.md](reduce/README.md)
- [maxsim/README.md](maxsim/README.md)
- [geospatial/README.md](geospatial/README.md)
- [mesh/README.md](mesh/README.md)
- [attention/README.md](attention/README.md), with GPU rows

## Related Projects

- [NumKong](https://github.com/ashvardanian/NumKong)
- [StringWars](https://github.com/ashvardanian/StringWars)

## License

Apache 2.0.
See [LICENSE](LICENSE).
