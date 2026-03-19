# NumWars

## Numerical Computing on CPUs, in Python & Rust

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
numkong::Tensor::dots_packed i8 → i32    ████████████████████ 1,357.36 GSO/s
numkong::Tensor::dots_packed bf16 → f32  ██████████▏            684.96 GSO/s
numkong::Tensor::dots_packed f16 → f32   █▋                     106.63 GSO/s
faer::linalg::matmul::matmul f32 → f32   █▎                      81.21 GSO/s
matrixmultiply::sgemm f32 → f32          █▏                      78.61 GSO/s
ndarray::ArrayBase::dot f32 → f32        █▏                      78.55 GSO/s
nalgebra::DMatrix × DMatrixᵀ f32 → f32   █▏                      74.21 GSO/s
numkong::Tensor::dots_packed f32 → f64   ▋                       42.04 GSO/s
```

Compared to Python:

```text
numkong.dots_packed i8 → i32    ████████████████████ 1,110.31 GSO/s
numkong.dots_packed bf16 → f32  ████████▊              487.89 GSO/s
numpy.matmul f32 → f32          ██▋                    145.73 GSO/s
numkong.dots_packed f16 → f32   █▋                      91.80 GSO/s
numkong.dots_packed f32 → f64   ▊                       42.69 GSO/s
```

See [dots/README.md](dots/README.md) for details.

### Pairwise Similarity

Single-pair vector kernels at 2048 dimensions.
This lists _Dot_ products and true _Euclidean_ distances measurements into one throughput-sorted view.
NumKong keeps its mixed-precision promotions, while the baseline libraries mostly stay in their input type.

Compared to Rust projects, it means:

```text
numkong::Dot::dot u8 → u32                ████████████████████ 54.28 GSO/s
numkong::Dot::dot i8 → i32                ███████████████▉     43.18 GSO/s
numkong::Euclidean::euclidean u8 → f32    ███████████████      40.83 GSO/s
numkong::Euclidean::euclidean i8 → f32    ████████████▋        34.10 GSO/s
numkong::Dot::dot bf16 → f32              ███████▍             20.09 GSO/s
scalar dot loop f32 → f32                 █████▎               14.25 GSO/s
numkong::Euclidean::euclidean bf16 → f32  ████▋                12.65 GSO/s
ndarray::ArrayBase::dot f32 → f32         ██▉                   7.75 GSO/s
nalgebra::Matrix::dot f32 → f32           ██▊                   7.56 GSO/s
scalar dot loop u8 → u32                  ██▋                   7.20 GSO/s
numkong::Dot::dot f32 → f64               ██▎                   6.12 GSO/s
numkong::Euclidean::euclidean f32 → f64   ██                    5.53 GSO/s
ndarray sqrt((a - b)·(a - b)) f32 → f32   █▊                    4.75 GSO/s
scalar dot loop i8 → i32                  █▊                    4.73 GSO/s
nalgebra (a - b).norm() f32 → f32         █▊                    4.63 GSO/s
scalar euclidean loop f32 → f32           ▋                     1.62 GSO/s
scalar euclidean loop u8 → f32            ▍                     1.18 GSO/s
scalar euclidean loop i8 → f32            ▍                     1.17 GSO/s
scalar euclidean loop bf16 → f32                                0.16 GSO/s
scalar dot loop bf16 → f32                                      0.16 GSO/s
```

Compared to Python:

```text
numkong.euclidean u8 → f32                  ████████████████████ 5.65 GSO/s
numkong.euclidean i8 → f32                  ██████████████████   5.08 GSO/s
numkong.dot u8 → u32                        █████████████████▎   4.88 GSO/s
numkong.euclidean f32 → f64                 ███████████▊         3.33 GSO/s
numkong.dot i8 → i32                        ███████████▌         3.25 GSO/s
scipy.linalg.blas.sdot f32 → f32            ███████████▏         3.14 GSO/s
numkong.dot f32 → f64                       █████████▊           2.76 GSO/s
scipy.spatial.distance.euclidean u8 → f32   █▊                   0.48 GSO/s
numkong.euclidean bf16 → f32                █▌                   0.41 GSO/s
scipy.spatial.distance.euclidean i8 → f32   █▍                   0.38 GSO/s
scipy.spatial.distance.euclidean f32 → f32  █▍                   0.38 GSO/s
numkong.dot bf16 → f32                      █▍                   0.37 GSO/s
```

See [similarity/README.md](similarity/README.md) for details.

### All-Pairs Similarity Matrices

Matrix-vs-matrix comparisons at 2048 rows by 2048 dimensions.
These are the packed many-to-many siblings of the pairwise spatial kernels above.
The merged lists below include _angular_ and _euclidean_ metrics, and the headline unit is GSO/s.

Compared to Rust projects, it means:

```text
numkong::Tensor::angulars_packed i8 → f32      ████████████████████ 590.88 GSO/s
numkong::Tensor::angulars_packed u8 → f32      ███████████████████▉ 588.37 GSO/s
numkong::Tensor::euclideans_packed u8 → f32    ███████████████████▊ 585.00 GSO/s
numkong::Tensor::euclideans_packed i8 → f32    ███████████████████▋ 581.99 GSO/s
numkong::Tensor::euclideans_packed bf16 → f32  ██████████▌          311.46 GSO/s
numkong::Tensor::angulars_packed bf16 → f32    ██████████▌          310.52 GSO/s
ndarray angular matrix f32 → f32               █▎                    37.15 GSO/s
ndarray euclidean matrix f32 → f32             █▏                    36.03 GSO/s
nalgebra euclidean matrix f32 → f32            █                     28.83 GSO/s
nalgebra angular matrix f32 → f32              █                     28.83 GSO/s
numkong::Tensor::angulars_packed f32 → f64     ▋                     19.44 GSO/s
numkong::Tensor::euclideans_packed f32 → f64   ▋                     19.37 GSO/s
```

Compared to Python through SciPy `cdist`:

```text
numkong.angulars_packed u8 → f32      ████████████████████ 465.04 GSO/s
numkong.euclideans_packed u8 → f32    ███████████████████▉ 463.47 GSO/s
numkong.euclideans_packed i8 → f32    ███████████████████▉ 463.37 GSO/s
numkong.angulars_packed i8 → f32      ███████████████████▌ 454.74 GSO/s
numkong.angulars_packed bf16 → f32    █████████▊           226.56 GSO/s
numkong.euclideans_packed bf16 → f32  █████████             210.12 GSO/s
numkong.euclideans_packed f32 → f64   ▉                     20.24 GSO/s
numkong.angulars_packed f32 → f64     ▊                     19.84 GSO/s
scipy.cdist euclidean f32 → f64       ▏                      2.83 GSO/s
scipy.cdist cosine f32 → f64          ▏                      2.62 GSO/s
```

See [similarities/README.md](similarities/README.md) for details.

### Elementwise Operations

Bandwidth-sensitive elementwise kernels (add, multiply, FMA) over 2048 elements.
FMA shown as representative sample.
In Rust:

```text
numkong::fma f32 → f32                  ████████████████████ 121.56 GB/s
scalar fma loop f32 → f32               ██████████████████▋  112.96 GB/s
ndarray fused multiply-add f32 → f32    ███████               42.30 GB/s
numkong::fma f16 → f16                  ███▊                  22.51 GB/s
numkong::fma bf16 → bf16                ███▎                  19.47 GB/s
```

In Python:

```text
numkong.fma f32 → f32                   ████████████████████ 1.24 GB/s
numkong.fma f16 → f16                   █████████▌           0.59 GB/s
numpy fma-style expression f32 → f32    ████▊                0.30 GB/s
numkong.fma bf16 → bf16                 █                    0.06 GB/s
numpy fma-style expression f16 → f16    ▏                    0.01 GB/s
```

See [each/README.md](each/README.md) for details.

### Reductions

Horizontal reductions over 2048 `f32` elements, including sum, norm, min/max, argmin/argmax, moments, minmax, and row_norms.
Sum shown as representative sample.
In Rust:

```text
polars::ChunkedArray::sum f32 → Option<f32>  ████████████████████ 43.86 GB/s
numkong::reduce_moments().sum f32 → f64      ███████████████████▊ 43.26 GB/s
ndarray::ArrayBase::sum f32 → f32            ████████████████▋    36.38 GB/s
scalar sum loop f32 → f32                    ███                   6.67 GB/s
```

See [reduce/README.md](reduce/README.md) for details.

### MaxSim

ColBERT-style late interaction with 32 query vectors, 128 document vectors, and 2048 dimensions.
NumKong promotes _f32 → f64_ here as well, while the baseline and matrix-style alternatives stay in _f32_.
In Rust:

```text
numkong::MaxSimPackedMatrix::score bf16 → f32  ████████████████████ 1,331.55 GSO/s
numkong::MaxSimPackedMatrix::score f16 → f32   ██████████████████▍  1,224.70 GSO/s
numkong::MaxSimPackedMatrix::score f32 → f64   ████████████▉          859.61 GSO/s
ndarray Q @ Dᵀ max-reduce f32 → f32            ▉                       57.87 GSO/s
nalgebra Q × Dᵀ max-reduce f32 → f32           ▉                       57.39 GSO/s
scalar MaxSim loop f32 → f32                                            3.26 GSO/s
```

See [maxsim/README.md](maxsim/README.md) for details.

### Geospatial Distances

Throughput over 2048 coordinate pairs.
The unit is MP/s, or million coordinate pairs per second.
The merged lists below include both _Haversine_ and _Vincenty_ distances.

Compared to Rust projects, it means:

```text
numkong::haversine       ████████████████████ 564.53 MP/s
numkong::vincenty        ██                    57.76 MP/s
geo::Haversine distance  ▉                     25.53 MP/s
geo::Vincenty distance                          1.20 MP/s
```

Compared to Python and its alternatives:

```text
numkong.haversine            ████████████████████ 526.05 MP/s
numkong.vincenty             ██▏                   57.22 MP/s
geopy.distance.great_circle                         0.21 MP/s
geopy.distance.geodesic                           0.0107 MP/s
```

See [geospatial/README.md](geospatial/README.md) for details.

### Mesh Alignment

Throughput over point clouds with 2048 3D points each.
The unit is MP/s, or million 3D points per second.
The labels include the full return signature so _RMSD_ and _Kabsch_ can share one sorted list cleanly.
In Rust:

```text
numkong::MeshAlignment::rmsd f32 → f64        ████████████████████ 610.32 MP/s
numkong::MeshAlignment::kabsch f32 → f64      ████████████▎        372.86 MP/s
nalgebra RMSD solve f32 → f32                 ██████▌              199.14 MP/s
nalgebra Kabsch solve f32 → f64               ████▏                125.14 MP/s
```

Compared to Python and its alternatives:

```text
numkong.rmsd f32 → f64                              ████████████████████ 468.79 MP/s
numkong.kabsch f32 → f64                            ███████████▏         260.75 MP/s
numkong.umeyama f32 → f64                           ██████████▍          245.37 MP/s
numpy rmsd f32 → f64                                ██▏                   51.06 MP/s
scipy Rotation.align_vectors (Kabsch) f32 → f64     ▌                     13.51 MP/s
biopython SVDSuperimposer (Kabsch) f32 → f64                              1.32 MP/s
```

See [mesh/README.md](mesh/README.md) for details.

## Replicating the Results

### Rust

```bash
RUSTFLAGS="-C target-cpu=native" \
NUMWARS_WARMUP_SECONDS=0.5 \
NUMWARS_PROFILE_SECONDS=2.0 \
NUMWARS_SAMPLE_SIZE=15 \
python scripts/update_root_readme.py
```

### Python

The generator creates and reuses a local `.venv` with `uv`, installs `.[similarity,each,dots,geospatial,mesh,reduce,similarities]`, and saves machine-readable outputs under `target/numwars/`.

To re-render the README without rerunning the benchmarks:

```bash
python scripts/update_root_readme.py --from-existing
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
