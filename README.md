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
> The numbers below are reference measurements collected on Apple M5 Pro (P-cores) in single-threaded mode.
> All benchmarks were run single-threaded on an idle system.
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
numkong::Tensor::dots_packed i8 → i32   ███████████████████████████████████ 2,783.00 GSO/s
numkong::Tensor::dots_packed bf16 → f32 ███████████████▋                    1,250.80 GSO/s
numkong::Tensor::dots_packed f16 → f32  ███████████████▋                    1,249.70 GSO/s
numkong::Tensor::dots_packed f32 → f64  ██▍                                   197.79 GSO/s

Alternatives:
nalgebra::DMatrix × DMatrixᵀ f32 → f32  █▍                                    118.72 GSO/s
ndarray::ArrayBase::dot f32 → f32       █▍                                    117.51 GSO/s
faer::linalg::matmul::matmul f32 → f32  █▍                                    117.50 GSO/s
matrixmultiply::sgemm f32 → f32         █▍                                    116.77 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.dots_packed i8 → i32   ████████████████████████████████████████████ 2,621.97 GSO/s
numkong.dots_packed bf16 → f32 ███████████████████▏                         1,142.19 GSO/s
numkong.dots_packed f16 → f32  ███████████████████                          1,134.69 GSO/s
numkong.dots_packed f32 → f64  ███▎                                           194.15 GSO/s

Alternatives:
numpy.matmul f32 → f32         ███████████████████████████████              1,854.27 GSO/s
```

See [dots/README.md](dots/README.md) for details.

### Pairwise Similarity

Single-pair vector kernels at 2048 dimensions.
This lists _Dot_ products and true _Euclidean_ distances measurements into one throughput-sorted view.
NumKong keeps its mixed-precision promotions, while the baseline libraries mostly stay in their input type.

Compared to Rust projects, it means:

```text
NumKong:
numkong::Dot::dot i8 → i32               █████████████████████████████████████ 37.41 GSO/s
numkong::Dot::dot u8 → u32               ████████████████████████████████████▉ 37.34 GSO/s
numkong::Euclidean::euclidean i8 → f32   ██████████████████████████████████▌   34.91 GSO/s
numkong::Euclidean::euclidean u8 → f32   ██████████████████████████████████▍   34.85 GSO/s
numkong::Dot::dot bf16 → f32             ██████████████▋                       14.83 GSO/s
numkong::Euclidean::euclidean bf16 → f32 █████▍                                 5.54 GSO/s
numkong::Dot::dot f32 → f64              ██▍                                    2.50 GSO/s
numkong::Euclidean::euclidean f32 → f64  ██▍                                    2.50 GSO/s

Alternatives:
nalgebra::Matrix::dot f32 → f32          ████████████▊                         12.95 GSO/s
ndarray::ArrayBase::dot f32 → f32        ████████████▋                         12.77 GSO/s
nalgebra (a - b).norm() f32 → f32        ███████▋                               7.74 GSO/s
ndarray sqrt((a - b)·(a - b)) f32 → f32  ███████▌                               7.61 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.euclidean i8 → f32  ██████████████████████████████████████████████████ 10.32 GSO/s
numkong.euclidean u8 → f32  █████████████████████████████████████████████████▌ 10.24 GSO/s
numkong.angular u8 → f32    ████████████████████████████████████████████████▊  10.07 GSO/s
numkong.angular i8 → f32    ████████████████████████████████████████████████▋  10.04 GSO/s
numkong.dot i8 → f32        ███████████████████████████████████████████████▌    9.81 GSO/s
numkong.dot u8 → f32        █████████████████████████████████████████████▍      9.37 GSO/s
numkong.dot f64 → f32       ████████████                                        2.48 GSO/s
numkong.euclidean f32 → f32 ██████████▍                                         2.15 GSO/s

Alternatives:
numpy.dot f32 → f32         ██████████████████▍                                 3.81 GSO/s
numpy.dot f64 → f64         █████████████████▊                                  3.68 GSO/s
```

See [similarity/README.md](similarity/README.md) for details.

### All-Pairs Similarity Matrices

Matrix-vs-matrix comparisons at 2048 rows by 2048 dimensions.
These are the packed many-to-many siblings of the pairwise spatial kernels above.
The merged lists below include _angular_ and _euclidean_ metrics, and the headline unit is GSO/s.

Compared to Rust projects, it means:

```text
NumKong:
numkong::Tensor::euclideans_packed u8 → f32   ███████████████████████████████ 888.74 GSO/s
numkong::Tensor::euclideans_packed i8 → f32   ██████████████████████████████▉ 887.85 GSO/s
numkong::Tensor::angulars_packed u8 → f32     ████████████████████████████▉   830.14 GSO/s
numkong::Tensor::angulars_packed i8 → f32     ████████████████████████████▉   830.13 GSO/s
numkong::Tensor::euclideans_packed bf16 → f32 ██████████████████▎             524.00 GSO/s
numkong::Tensor::angulars_packed bf16 → f32   █████████████████▌              502.45 GSO/s
numkong::Tensor::euclideans_packed f32 → f64  ███▏                             92.93 GSO/s
numkong::Tensor::angulars_packed f32 → f64    ███▏                             92.52 GSO/s

Alternatives:
ndarray euclidean matrix f32 → f32            ██                               57.64 GSO/s
ndarray angular matrix f32 → f32              █▉                               56.98 GSO/s
nalgebra angular matrix f32 → f32             █▋                               49.95 GSO/s
nalgebra euclidean matrix f32 → f32           █▋                               49.79 GSO/s
```

Compared to Python through SciPy `cdist`:

```text
NumKong:
numkong.euclideans_packed u8 → f32  █████████████████████████████████████████ 425.91 GSO/s
numkong.euclideans_packed i8 → f32  ███████████████████████████████████████▎  408.64 GSO/s
numkong.angulars_packed i8 → f32    █████████████████████████████████████▎    386.96 GSO/s
numkong.angulars_packed u8 → f32    ███████████████████████████████████       364.01 GSO/s
numkong.angulars_packed f32 → f64   ███████▋                                   79.26 GSO/s
numkong.euclideans_packed f32 → f64 █████                                      52.95 GSO/s

Alternatives:
scipy.cdist euclidean f32 → f64     ▍                                           5.09 GSO/s
scipy.cdist cosine f32 → f64        ▏                                           1.30 GSO/s
```

See [similarities/README.md](similarities/README.md) for details.

### Elementwise Operations

Bandwidth-sensitive elementwise kernels — add and scale — over 1,000,000 elements.
Sum shown as representative sample.
In Rust:

```text
NumKong:
numkong::EachSum i8 → i8   ███████████████████████████████████████████████████ 111.47 GB/s
numkong::EachSum f32 → f32 ████████████████████████████████████████████▋        97.55 GB/s
numkong::EachSum f16 → f16 ████████████████████████████████████████████▏        96.56 GB/s

Alternatives:
nalgebra::add f32 → f32    ███████████████████████████████████████████▌         95.31 GB/s
ndarray::add f32 → f32     ███████████████████████████████████████████▍         94.84 GB/s
serial code f32 → f32      ███████████████████████████████████████████          94.06 GB/s
```

In Python:

```text
numpy.add i8 → i8       ██████████████████████████████████████████████████████ 143.56 GB/s
numkong.add i8 → i8     ██████████████████████████████████████████████▌        123.77 GB/s
numkong.add f32 → f32   ████████████████████████████████████████████▌          118.39 GB/s
numpy.add f32 → f32     ███████████████████████████████████████████▍           115.32 GB/s
numpy.add f64 → f64     ███████████████████████████████████████████            114.37 GB/s
numkong.add f16 → f16   ████████████████████████████████████████▎              107.29 GB/s
numkong.add f64 → f64   █████████████████████████████████████▌                 100.01 GB/s
numkong.add bf16 → bf16 ███████████████████████████▌                            73.27 GB/s
numpy.add f16 → f16     █▌                                                       4.08 GB/s
```

See [each/README.md](each/README.md) for details.

### Reductions

Horizontal reductions over 1,000,000 elements.
The suite covers sum and row-wise L2 norms.
In Rust:

```text
polars::ChunkedArray::sum f64 → f64     ██████████████████████████████████████ 113.57 GB/s
polars::ChunkedArray::sum f32 → f32     █████████████████████████████████████  110.70 GB/s
ndarray::ArrayBase::sum f64 → f64       █████████████████████████████████▎      99.49 GB/s
ndarray::ArrayBase::sum f32 → f32       ████████████████▋                       49.83 GB/s
numkong::reduce_moments().sum f32 → f64 ███▍                                    10.31 GB/s
serial sum loop f32 → f32               ██▊                                      8.50 GB/s
```

Row-wise L2 norms over a 2048×2048 matrix:

```text
ndarray row norms f64 → f64             ███████████████████████████████████████ 89.72 GB/s
ndarray row norms f32 → f32             ███████████████████████▏                53.24 GB/s
numkong::Dot self-dot + sqrt bf16 → f32 █████████████▎                          30.64 GB/s
numkong::Dot self-dot + sqrt f64        ██████████▏                             23.44 GB/s
serial row norms loop f64 → f64         ███████▊                                17.95 GB/s
numkong::Dot self-dot + sqrt f32        ████▌                                   10.60 GB/s
serial row norms loop f32 → f32         ███▉                                     9.20 GB/s
```

In Python over 1,000,000 elements:

```text
numpy.sum f64 → f64         ███████████████████████████████████████████████████ 61.26 GB/s
numpy.sum f32 → f32         ████████████████████████████▏                       33.92 GB/s
numpy.linalg.norm f64 → f64 █████████████████████████▏                          30.26 GB/s
numkong.sum u8 → u8         ██████████████████▏                                 21.78 GB/s
numkong.sum i8 → i8         █████████████████▊                                  21.40 GB/s
numpy.linalg.norm f32 → f64 ████████████████▊                                   20.15 GB/s
numkong.norm f64 → f64      ██████████████▌                                     17.44 GB/s
numkong.sum f64 → f64       █████████████▌                                      16.34 GB/s
numkong.norm f32 → f64      ████████████▌                                       15.10 GB/s
numkong.sum f32 → f32       ███████▉                                             9.49 GB/s
numpy.sum i8 → i8           █████▌                                               6.73 GB/s
```

See [reduce/README.md](reduce/README.md) for details.

### MaxSim

ColBERT-style late interaction with 2048 query vectors, 2048 document vectors, and 2048 dimensions.
NumKong promotes _f32 → f64_ here as well, while ndarray stays in _f32_.
In Rust:

```text
NumKong:
numkong::MaxSimPackedMatrix::score f32 → f64  █████████████████████████████ 1,483.41 GSO/s
numkong::MaxSimPackedMatrix::score bf16 → f32 ███████████████████▏            983.57 GSO/s
numkong::MaxSimPackedMatrix::score f16 → f32  ███████████████████▏            980.33 GSO/s

Alternatives:
ndarray Q @ Dᵀ max-reduce f32 → f32           █▏                               58.37 GSO/s
```

Compared to Python:

```text
NumKong:
numkong.maxsim_packed f32 → f64   ██████████████████████████████████████████ 2,425.72 GSO/s
numkong.maxsim_packed bf16 → f32  █████████████████████▍                     1,236.30 GSO/s
numkong.maxsim_packed f16 → f32   ████████████                                696.78 GSO/s

Alternatives:
numpy matmul f32 → f32            ██████████████████████████▍                1,525.56 GSO/s
```

See [maxsim/README.md](maxsim/README.md) for details.

### Geospatial Distances

Throughput over 2048 coordinate pairs.
The unit is MP/s, or million coordinate pairs per second.
The merged lists below include both _Haversine_ and _Vincenty_ distances.

Compared to Rust projects, it means:

```text
NumKong:
numkong::haversine f32 → f32      ████████████████████████████████████████████ 491.98 MP/s
numkong::haversine f64 → f64      █████████████▍                               149.72 MP/s
numkong::vincenty f32 → f32       ██████▍                                       71.64 MP/s
numkong::vincenty f64 → f64       █▏                                            13.73 MP/s

Alternatives:
geo::Haversine distance f32 → f32 ████████████▏                                136.96 MP/s
geo::Haversine distance f64 → f64 ████████▎                                     92.48 MP/s
geo::Vincenty distance f64 → f64  ▏                                              2.76 MP/s
```

Compared to Python and its alternatives:

```text
NumKong:
numkong.haversine f32 → f32           ████████████████████████████████████████ 444.38 MP/s
numkong.haversine f64 → f64           ███████████▉                             132.85 MP/s
numkong.vincenty f32 → f32            █████▉                                    65.89 MP/s
numkong.vincenty f64 → f64            █                                         11.93 MP/s

Alternatives:
geopy.distance.great_circle f64 → f64                                            0.47 MP/s
geopy.distance.geodesic f64 → f64                                                0.03 MP/s
```

See [geospatial/README.md](geospatial/README.md) for details.

### Mesh Alignment

Throughput over point clouds with 2048 3D points each.
The unit is MP/s, or million 3D points per second.
The labels include the full return signature so _RMSD_ and _Kabsch_ can share one sorted list cleanly.
In Rust:

```text
NumKong:
numkong::MeshAlignment::rmsd f16 → f16      ████████████████████████████████ 2,864.47 MP/s
numkong::MeshAlignment::rmsd bf16 → bf16    ███████████████████████████████▉ 2,861.70 MP/s
numkong::MeshAlignment::rmsd f64 → f64      ████████████████████▊            1,859.32 MP/s
numkong::MeshAlignment::rmsd f32 → f32      ██████████████████▏              1,626.67 MP/s
numkong::MeshAlignment::kabsch f16 → f16    ███████▊                           696.00 MP/s
numkong::MeshAlignment::kabsch bf16 → bf16  ███████▋                           691.01 MP/s
numkong::MeshAlignment::umeyama bf16 → bf16 ███████▌                           673.50 MP/s
numkong::MeshAlignment::umeyama f16 → f16   ██████▊                            614.06 MP/s
numkong::MeshAlignment::kabsch f32 → f32    ████▍                              396.52 MP/s
numkong::MeshAlignment::umeyama f32 → f32   ████▏                              376.48 MP/s
numkong::MeshAlignment::kabsch f64 → f64    ███▋                               331.70 MP/s
numkong::MeshAlignment::umeyama f64 → f64   ███▋                               325.16 MP/s

Alternatives:
nalgebra-based RMSD f32 → f32               ███████                            634.04 MP/s
nalgebra-based Kabsch f32 → f64             ███▏                               283.16 MP/s
nalgebra-based Umeyama f32 → f64            ██▊                                255.14 MP/s
```

Compared to Python and its alternatives:

```text
NumKong:
numkong.rmsd f64 → f64                       ███████████████████████████████ 1,311.77 MP/s
numkong.rmsd f32 → f64                       █████████████████████████████   1,228.00 MP/s
numkong.kabsch f32 → f64                     ████████▌                         360.08 MP/s
numkong.umeyama f32 → f64                    ███████▋                          327.01 MP/s
numkong.umeyama f64 → f64                    ███████                           296.67 MP/s
numkong.kabsch f64 → f64                     ██████▊                           285.81 MP/s

Alternatives:
numpy-based RMSD f32 → f64                   ██▉                               124.48 MP/s
numpy-based RMSD f64 → f64                   ██▊                               117.30 MP/s
biopython SVDSuperimposer (Kabsch) f32 → f64                                     2.88 MP/s
biopython SVDSuperimposer (Kabsch) f64 → f64                                     2.92 MP/s
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
