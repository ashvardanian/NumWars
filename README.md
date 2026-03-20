# NumWars

## Mixed-Precision Numerics Benchmarks for Rust & Python

![NumWars banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/NumWars-v1.png?raw=true)

There are many strong libraries for numerical computing.
Most of them are written in C, C++, and Fortran, with excellent Rust wrappers and Python bindings on top.

Where Rust is especially convenient is dependency management and reproducible benchmarking, making it a good place to line up apples-to-apples comparisons across native crates and their Python bindings.
NumWars exists for the same reason StringWars exists for StringZilla: to compare [NumKong](https://github.com/ashvardanian/NumKong) against mainstream CPU stacks on the workloads it was built for, including:

- [`ndarray`](https://github.com/rust-ndarray/ndarray) and [`nalgebra`](https://github.com/dimforge/nalgebra) for dense tensor and linear algebra kernels.
- [`faer`](https://github.com/sarah-quinones/faer-rs) and [`matrixmultiply`](https://github.com/bluss/matrixmultiply) for GEMM-like Rust baselines.
- [`geo`](https://github.com/georust/geo) for geographic distances.
- [`polars`](https://github.com/pola-rs/polars) and reduction-heavy analytics workloads.
- [`NumPy`](https://github.com/numpy/numpy), [`SciPy`](https://github.com/scipy/scipy), and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) on Python.

Of course, the APIs and internal kernels of those projects are different.
So this repository focuses on the workload families NumKong was designed for and compares their effective throughput using the native unit for each operation family instead of forcing everything into fake global `ops/s`.

> [!IMPORTANT]
> The numbers below are reference measurements collected on Intel Sapphire Rapids CPU in single-threaded mode.
> They will vary with CPU model, compiler flags, BLAS backend, and problem size.
> Rebuild and rerun on your own hardware before treating them as absolute.

## Benchmarks at a Glance

### Packed Matrix Multiplication

NumKong packed dots are mixed-precision by design.
_i8_ inputs produce _i32_ outputs.
_bf16_ and _f16_ inputs produce _f32_ outputs.
_f32_ inputs produce _f64_ outputs.
The mainstream baselines shown here keep _f32 → f32_.
Compared to Rust projects, it means:

```text
NumKong:
numkong::Tensor::dots_packed i8 → i32    ██████████████████████████████████ 1,357.36 GSO/s
numkong::Tensor::dots_packed bf16 → f32  █████████████████▎                   684.96 GSO/s
numkong::Tensor::dots_packed f16 → f32   ██▊                                  106.63 GSO/s
numkong::Tensor::dots_packed f32 → f64   █                                     42.04 GSO/s

Alternatives:
faer::linalg::matmul::matmul f32 → f32   ██▏                                   81.21 GSO/s
matrixmultiply::sgemm f32 → f32          █▉                                    78.61 GSO/s
ndarray::ArrayBase::dot f32 → f32        █▉                                    78.55 GSO/s
nalgebra::DMatrix × DMatrixᵀ f32 → f32   █▉                                    74.21 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.dots_packed i8 → i32    ███████████████████████████████████████████ 1,110.31 GSO/s
numkong.dots_packed bf16 → f32  ██████████████████▊                           487.89 GSO/s
numkong.dots_packed f16 → f32   ███▌                                           91.80 GSO/s
numkong.dots_packed f32 → f64   █▋                                             42.69 GSO/s

Alternatives:
numpy.matmul f32 → f32          █████▋                                        145.73 GSO/s
```

See [dots/README.md](dots/README.md) for details.

### Pairwise Similarity

Single-pair vector kernels at 2048 dimensions.
This lists _Dot_ products and true _Euclidean_ distances measurements into one throughput-sorted view.
NumKong keeps its mixed-precision promotions, while the baseline libraries mostly stay in their input type.

Compared to Rust projects, it means:

```text
NumKong:
numkong::Dot::dot u8 → u32                ████████████████████████████████████ 54.28 GSO/s
numkong::Dot::dot i8 → i32                ████████████████████████████▋        43.18 GSO/s
numkong::Euclidean::euclidean u8 → f32    ███████████████████████████          40.83 GSO/s
numkong::Euclidean::euclidean i8 → f32    ██████████████████████▊              34.10 GSO/s
numkong::Dot::dot bf16 → f32              █████████████▎                       20.09 GSO/s
numkong::Euclidean::euclidean bf16 → f32  ████████▍                            12.65 GSO/s
numkong::Dot::dot f32 → f64               ████                                  6.12 GSO/s
numkong::Euclidean::euclidean f32 → f64   ███▋                                  5.53 GSO/s

Alternatives:
ndarray::ArrayBase::dot f32 → f32         █████▏                                7.75 GSO/s
nalgebra::Matrix::dot f32 → f32           █████                                 7.56 GSO/s
ndarray sqrt((a - b)·(a - b)) f32 → f32   ███▏                                  4.75 GSO/s
nalgebra (a - b).norm() f32 → f32         ███▏                                  4.63 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.euclidean u8 → f32                  ███████████████████████████████████ 5.65 GSO/s
numkong.euclidean i8 → f32                  ███████████████████████████████▌    5.08 GSO/s
numkong.dot u8 → u32                        ██████████████████████████████▎     4.88 GSO/s
numkong.euclidean f32 → f64                 ████████████████████▌               3.33 GSO/s
numkong.dot i8 → i32                        ████████████████████▏               3.25 GSO/s
numkong.dot f32 → f64                       █████████████████                   2.76 GSO/s
numkong.euclidean bf16 → f32                ██▋                                 0.41 GSO/s
numkong.dot bf16 → f32                      ██▍                                 0.37 GSO/s

Alternatives:
scipy.linalg.blas.sdot f32 → f32            ███████████████████▌                3.14 GSO/s
scipy.spatial.distance.euclidean u8 → f32   ███                                 0.48 GSO/s
scipy.spatial.distance.euclidean i8 → f32   ██▍                                 0.38 GSO/s
scipy.spatial.distance.euclidean f32 → f32  ██▍                                 0.38 GSO/s
```

See [similarity/README.md](similarity/README.md) for details.

### All-Pairs Similarity Matrices

Matrix-vs-matrix comparisons at 2048 rows by 2048 dimensions.
These are the packed many-to-many siblings of the pairwise spatial kernels above.
The merged lists below include _angular_ and _euclidean_ metrics, and the headline unit is GSO/s.

Compared to Rust projects, it means:

```text
NumKong:
numkong::Tensor::angulars_packed u8 → f32      ██████████████████████████████ 694.88 GSO/s
numkong::Tensor::angulars_packed i8 → f32      █████████████████████████████▋ 686.93 GSO/s
numkong::Tensor::euclideans_packed i8 → f32    █████████████████████████████▋ 685.67 GSO/s
numkong::Tensor::euclideans_packed u8 → f32    █████████████████████████████  672.37 GSO/s
numkong::Tensor::angulars_packed bf16 → f32    █████████████▏                 304.59 GSO/s
numkong::Tensor::euclideans_packed bf16 → f32  █████████████                  302.61 GSO/s
numkong::Tensor::euclideans_packed f32 → f64   █                               21.22 GSO/s
numkong::Tensor::angulars_packed f32 → f64     █                               20.64 GSO/s

Alternatives:
ndarray angular matrix f32 → f32               █▊                              38.20 GSO/s
nalgebra euclidean matrix f32 → f32            █▊                              37.91 GSO/s
ndarray euclidean matrix f32 → f32             █▊                              37.59 GSO/s
nalgebra angular matrix f32 → f32              █▊                              36.97 GSO/s
```

Compared to Python through SciPy `cdist`:

```text
NumKong:
numkong.angulars_packed u8 → f32      ███████████████████████████████████████ 465.04 GSO/s
numkong.euclideans_packed u8 → f32    ██████████████████████████████████████▊ 463.47 GSO/s
numkong.euclideans_packed i8 → f32    ██████████████████████████████████████▊ 463.37 GSO/s
numkong.angulars_packed i8 → f32      ██████████████████████████████████████  454.74 GSO/s
numkong.angulars_packed bf16 → f32    ███████████████████                     226.56 GSO/s
numkong.euclideans_packed bf16 → f32  █████████████████▌                      210.12 GSO/s
numkong.euclideans_packed f32 → f64   █▊                                       20.24 GSO/s
numkong.angulars_packed f32 → f64     █▌                                       19.84 GSO/s

Alternatives:
scipy.cdist euclidean f32 → f64       ▎                                         2.83 GSO/s
scipy.cdist cosine f32 → f64          ▎                                         2.62 GSO/s
```

See [similarities/README.md](similarities/README.md) for details.

### Elementwise Operations

Bandwidth-sensitive elementwise kernels — add and scale — over 1,000,000 elements.
Sum shown as representative sample.
In Rust:

```text
NumKong:
numkong::EachSum i8 → i8      █████████████████████████████████████████████     23.23 GB/s
numkong::EachSum f16 → f16    ██████████████████████████████████████▌           19.93 GB/s
numkong::EachSum bf16 → bf16  ████████████████████████████████████▉             19.08 GB/s
numkong::EachSum f32 → f32    ████████████████████████████████████▎             18.73 GB/s

Alternatives:
serial code f32 → f32         ██████████████████████████████████████▍           19.86 GB/s
nalgebra::add f32 → f32       ██████████████████████████████████████▍           19.82 GB/s
ndarray::add f32 → f32        ██████████████████████████████████████▎           19.79 GB/s
```

In Python:

```text
NumKong:
numkong.add i8 → i8      █████████████████████████████████████████▋             30.91 GB/s
numkong.add f32 → f32    ███████████████████████████████████████▋               29.39 GB/s
numkong.add f16 → f16    ██████████████████████████████████████▉                28.84 GB/s
numkong.add f64 → f64    ██████████████████████████████████████▉                28.79 GB/s
numkong.add bf16 → bf16  █████████████████████████████████████▍                 27.72 GB/s

Alternatives:
numpy.add i8 → i8        █████████████████████████████████████████████          33.32 GB/s
numpy.add f32 → f32      ██████████████████████████████████▋                    25.65 GB/s
numpy.add f64 → f64      █████████████████████████████████▊                     25.03 GB/s
numpy.add f16 → f16      █▎                                                      0.95 GB/s
```

See [each/README.md](each/README.md) for details.

### Reductions

Horizontal reductions over 1,000,000 elements.
The suite covers sum and row-wise L2 norms.
In Rust:

```text
ndarray::ArrayBase::sum f32 → f32        █████████████████████████████████████████████ 32.50 GB/s
polars::ChunkedArray::sum f32 → f32      ███████████████████████████████████████████▎ 31.26 GB/s
numkong::reduce_moments().sum f32 → f64  █████████████████████████████████████████▋ 30.09 GB/s
serial sum loop f32 → f32                ████████▊                               6.38 GB/s
```

Row-wise L2 norms over a 2048×2048 matrix:

```text
ndarray row norms f64 → f64              █████████████████████████████████████████████ 27.11 GB/s
numkong::Dot self-dot + sqrt bf16 → f32  ████████████████████████████████████████▌ 24.46 GB/s
ndarray row norms f32 → f32              ███████████████████████████████████▉   21.63 GB/s
numkong::Dot self-dot + sqrt f32         █████████████████████████████████▏     20.01 GB/s
serial row norms loop f32 → f32          ██████████▊                             6.54 GB/s
```

In Python over 1,000,000 elements:

```text
NumKong:
numkong.sum i8 → i8          █████████████████████████████████████████████      32.02 GB/s
numkong.sum f32 → f32        ████████████████████████████████████████▉          29.17 GB/s
numkong.norm f32 → f64       ███████████████████████████████▎                   22.32 GB/s
numkong.sum f64 → f64        █████████████████████████████                      20.68 GB/s
numkong.norm bf16 → f64      █████████████████████████                          17.82 GB/s

Alternatives:
numpy.sum f64 → f64          █████████████████████████████████▉                 24.16 GB/s
numpy.sum f32 → f32          ██████████████████████████▊                        19.06 GB/s
numpy.linalg.norm f64 → f64  ███████████▌                                        8.21 GB/s
numpy.linalg.norm f32 → f64  ██████████▌                                         7.48 GB/s
numpy.sum i8 → i8            ███▊                                                2.68 GB/s
```

See [reduce/README.md](reduce/README.md) for details.

### MaxSim

ColBERT-style late interaction with 2048 query vectors, 2048 document vectors, and 2048 dimensions.
NumKong promotes _f32 → f64_ here as well, while ndarray stays in _f32_.
In Rust:

```text
NumKong:
numkong::MaxSimPackedMatrix::score f16 → f32   ██████████████████████████████ 423.69 GSO/s
numkong::MaxSimPackedMatrix::score f32 → f64   █████████████████████████████▌ 415.47 GSO/s
numkong::MaxSimPackedMatrix::score bf16 → f32  ███████████████▊               224.48 GSO/s

Alternatives:
ndarray Q @ Dᵀ max-reduce f32 → f32            ██▋                             38.36 GSO/s
```

See [maxsim/README.md](maxsim/README.md) for details.

### Geospatial Distances

Throughput over 2048 coordinate pairs.
The unit is MP/s, or million coordinate pairs per second.
The merged lists below include both _Haversine_ and _Vincenty_ distances.

Compared to Rust projects, it means:

```text
NumKong:
numkong::haversine f32 → f32      ████████████████████████████████████████████ 486.92 MP/s
numkong::haversine f64 → f64      █████████████▊                               151.65 MP/s
numkong::vincenty f32 → f32       ██████▍                                       68.96 MP/s
numkong::vincenty f64 → f64       █▋                                            17.79 MP/s

Alternatives:
geo::Haversine distance f32 → f32 ███▋                                          38.88 MP/s
geo::Haversine distance f64 → f64 ██▎                                           24.07 MP/s
geo::Vincenty distance f64 → f64  ▏                                              1.15 MP/s
```

Compared to Python and its alternatives:

```text
NumKong:
numkong.haversine f32 → f32           ████████████████████████████████████████ 475.41 MP/s
numkong.haversine f64 → f64           █████████████                            154.92 MP/s
numkong.vincenty f32 → f32            ████▊                                     54.99 MP/s
numkong.vincenty f64 → f64            █▌                                        17.87 MP/s

Alternatives:
geopy.distance.great_circle f64 → f64                                            0.18 MP/s
geopy.distance.geodesic f64 → f64                                              0.0096 MP/s
```

See [geospatial/README.md](geospatial/README.md) for details.

### Mesh Alignment

Throughput over point clouds with 2048 3D points each.
The unit is MP/s, or million 3D points per second.
The labels include the full return signature so _RMSD_ and _Kabsch_ can share one sorted list cleanly.
In Rust:

```text
NumKong:
numkong::MeshAlignment::rmsd f64 → f64      ██████████████████████████████████ 971.35 MP/s
numkong::MeshAlignment::rmsd f32 → f32      ████████████████████▊              592.73 MP/s
numkong::MeshAlignment::rmsd f16 → f16      ████████████████████▎              578.61 MP/s
numkong::MeshAlignment::rmsd bf16 → bf16    ███████████████████▉               567.69 MP/s
numkong::MeshAlignment::kabsch f32 → f32    ██████████████▏                    404.69 MP/s
numkong::MeshAlignment::umeyama f32 → f32   ███████████▊                       335.03 MP/s
numkong::MeshAlignment::kabsch bf16 → bf16  █████████▌                         272.09 MP/s
numkong::MeshAlignment::umeyama bf16 → bf16 █████████▍                         268.63 MP/s
numkong::MeshAlignment::kabsch f16 → f16    █████████▎                         264.46 MP/s
numkong::MeshAlignment::umeyama f16 → f16   █████████▎                         264.89 MP/s
numkong::MeshAlignment::kabsch f64 → f64    ████████▋                          245.90 MP/s
numkong::MeshAlignment::umeyama f64 → f64   █████▏                             147.75 MP/s

Alternatives:
nalgebra-based RMSD f32 → f32               ██████████████████▊                537.04 MP/s
nalgebra-based Kabsch f32 → f64             ████▎                              121.63 MP/s
nalgebra-based Umeyama f32 → f64            ███▊                               106.47 MP/s
```

Compared to Python and its alternatives:

```text
NumKong:
numkong.rmsd f64 → f64                       █████████████████████████████████ 825.51 MP/s
numkong.rmsd f32 → f64                       ██████████████████▋               467.35 MP/s
numkong.kabsch f32 → f64                     █████████▉                        248.48 MP/s
numkong.umeyama f32 → f64                    █████████▉                        248.10 MP/s
numkong.kabsch f64 → f64                     █████████▌                        238.79 MP/s
numkong.umeyama f64 → f64                    ██████▍                           159.25 MP/s

Alternatives:
numpy-based RMSD f32 → f64                   ██▏                                50.49 MP/s
numpy-based RMSD f64 → f64                   █▉                                 46.74 MP/s
biopython SVDSuperimposer (Kabsch) f32 → f64                                     1.22 MP/s
biopython SVDSuperimposer (Kabsch) f64 → f64                                     1.19 MP/s
```

See [mesh/README.md](mesh/README.md) for details.

## Replicating the Results

### Rust

Every Rust benchmark is a Criterion harness behind a Cargo feature gate.
Run one suite at a time or all at once:

```bash
# One suite — default 2048-element workload
RUSTFLAGS="-C target-cpu=native" \
cargo bench --features bench_similarity --bench bench_similarity

# All suites
RUSTFLAGS="-C target-cpu=native" \
cargo bench --features all
```

Tuning knobs (environment variables):

| Variable                  | Default  | Purpose                                           |
| :------------------------ | :------- | :------------------------------------------------ |
| `NUMWARS_DIMS`            | 2048     | Vector / matrix dimension shared by most suites   |
| `NUMWARS_DIMS_HEIGHT`     | 2048     | Row count for GEMM workloads (dots, maxsim)       |
| `NUMWARS_DIMS_WIDTH`      | 2048     | Column count for GEMM workloads (dots, maxsim)    |
| `NUMWARS_DIMS_DEPTH`      | 2048     | Shared (contraction) dimension for GEMM workloads |
| `NUMWARS_FILTER`          | _(none)_ | Regex to select benchmarks by name                |
| `NUMWARS_WARMUP_SECONDS`  | 3.0      | Criterion warm-up time                            |
| `NUMWARS_PROFILE_SECONDS` | 10.0     | Criterion measurement time                        |
| `NUMWARS_SAMPLE_SIZE`     | 50       | Criterion sample count                            |

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
- [similarities/README.md](similarities/README.md)
- [dots/README.md](dots/README.md)
- [each/README.md](each/README.md)
- [reduce/README.md](reduce/README.md)
- [maxsim/README.md](maxsim/README.md)
- [geospatial/README.md](geospatial/README.md)
- [mesh/README.md](mesh/README.md)

## Related Projects

- [NumKong](https://github.com/ashvardanian/NumKong)
- [StringWars](https://github.com/ashvardanian/StringWars)

## License

Apache 2.0.
See [LICENSE](LICENSE).
