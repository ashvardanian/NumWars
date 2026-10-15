# Mesh Benchmarks

Many-to-many shape distance benchmarks between 3D point clouds, comparing NumKong against nalgebra, lapjv, SciPy, and POT.
Every call scores 8 row clouds against 8 column clouds of 32 points each, with the correspondence-free distances that replaced mesh alignment: _Chamfer_, _Hausdorff_, _Wasserstein-Procrustes_ and _Gromov-Wasserstein_.
Clouds pack once outside the timed loop, centered and rotated onto their principal axes, and every distance takes the best of the 8 sign flips of those axes.
The baselines pack the same way and search the same flips.
Between clouds of one size the exact transport plan is an assignment, so the Wasserstein-Procrustes baselines alternate an optimal assignment with a Procrustes fit, through `lapjv` and nalgebra's SVD in Rust and SciPy's `linear_sum_assignment` and `orthogonal_procrustes` in Python.
Gromov-Wasserstein has no Rust competitor, and POT's `gromov_wasserstein2` stands in for it in Python.
The unit is MP/s, or million cloud pairs per second.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong widens _f16_ inputs to _f32_ outputs.
The `baseline_*` alternatives are the hand-written nalgebra and lapjv loops of `bench.rs`.

## Intel Xeon5 Emerald Rapids

### Rust

| NumKong                                                           |         Speed | Speedup | Alternative                                    |        Speed |
| :---------------------------------------------------------------- | ------------: | ------: | :--------------------------------------------- | -----------: |
| `numkong::ShapesPackedMatrix::chamfers`, _f32 → f64_              |    0.627 MP/s |   7.31× | `baseline_distance`                            |  0.0858 MP/s |
| `numkong::ShapesPackedMatrix::chamfers`, _f16 → f32_              |    0.978 MP/s |   11.4× | `baseline_distance`, _f16 → f64_               |  0.0857 MP/s |
| `numkong::ShapesPackedMatrix::hausdorffs`, _f32 → f64_            |    0.626 MP/s |   7.30× | `baseline_distance`                            |  0.0858 MP/s |
| `numkong::ShapesPackedMatrix::hausdorffs`, _f16 → f32_            |    0.981 MP/s |   11.4× | `baseline_distance`, _f16 → f64_               |  0.0859 MP/s |
| `numkong::ShapesPackedMatrix::wassersteinprocrustes`, _f32 → f64_ |  0.00231 MP/s |   0.83× | `baseline_wasserstein_procrustes`              | 0.00277 MP/s |
| `numkong::ShapesPackedMatrix::wassersteinprocrustes`, _f16 → f32_ |  0.00210 MP/s |   0.76× | `baseline_wasserstein_procrustes`, _f16 → f64_ | 0.00276 MP/s |
| `numkong::ShapesPackedMatrix::gromovwassersteins`, _f32 → f64_    | 0.000933 MP/s |       – | –                                              |            – |
| `numkong::ShapesPackedMatrix::gromovwassersteins`, _f16 → f32_    | 0.000837 MP/s |       – | –                                              |            – |

> Measured October 14, 2026.

### Python

| NumKong                                             |         Speed | Speedup | Alternative                                         |         Speed |
| :-------------------------------------------------- | ------------: | ------: | :-------------------------------------------------- | ------------: |
| `numkong.chamfers_packed`, _f32 → f64_              |     0.62 MP/s |    174× | `scipy.spatial.cKDTree`                             |  0.00357 MP/s |
| `numkong.chamfers_packed`, _f16 → f32_              |     0.96 MP/s |    275× | `scipy.spatial.cKDTree`, _f16 → f64_                |  0.00349 MP/s |
| `numkong.hausdorffs_packed`, _f32 → f64_            |     0.61 MP/s |    158× | `scipy.spatial.cKDTree`                             |  0.00385 MP/s |
| `numkong.hausdorffs_packed`, _f16 → f32_            |     0.97 MP/s |    256× | `scipy.spatial.cKDTree`, _f16 → f64_                |  0.00379 MP/s |
| `numkong.wassersteinprocrustes_packed`, _f32 → f64_ |  0.00226 MP/s |   4.33× | `scipy.optimize.linear_sum_assignment`              | 0.000522 MP/s |
| `numkong.wassersteinprocrustes_packed`, _f16 → f32_ |  0.00202 MP/s |   3.79× | `scipy.optimize.linear_sum_assignment`, _f16 → f64_ | 0.000533 MP/s |
| `numkong.gromovwassersteins_packed`, _f32 → f64_    | 0.000895 MP/s |   0.63× | `ot.gromov.gromov_wasserstein2`                     |  0.00143 MP/s |
| `numkong.gromovwassersteins_packed`, _f16 → f32_    | 0.000905 MP/s |   0.62× | `ot.gromov.gromov_wasserstein2`, _f16 → f64_        |  0.00146 MP/s |

> Measured October 14, 2026.

## Run It

### Rust

```bash
# Default 8×8 clouds of 32 points
cargo bench --bench bench_mesh --features bench_mesh

# Larger 128-point clouds
NUMWARS_SHAPE_POINTS=128 cargo bench --bench bench_mesh --features bench_mesh

# Focus on one distance
NUMWARS_FILTER="mesh/chamfers" \
cargo bench --bench bench_mesh --features bench_mesh
```

### Python

```bash
# Default 8×8 clouds of 32 points
python mesh/bench.py

# Larger 128-point clouds
NUMWARS_SHAPE_POINTS=128 python mesh/bench.py

# Focus on one distance
NUMWARS_FILTER="mesh/chamfers" python mesh/bench.py
```
