# Geospatial Benchmarks

Many-to-many geospatial distance benchmarks comparing NumKong's packed kernels against `geo`, ndarray, scikit-learn, pyproj, SciPy, NumPy, and scalar baselines.
Every call scores each of 1024 row points against each of 128 column points into a row-major matrix, and each point set packs once outside the timed loop, as the dots suite packs its B operand.
_Haversine_ distances run on the IUGG mean sphere, and _Vincenty_ distances on the WGS-84 ellipsoid, where `geo` and pyproj answer with Karney's geodesic solver instead.
_Squared Euclidean_ distances between Earth-fixed points take _i32_ centimeters, exact in _u64_, or _f64_ meters.
The unit is MP/s, or million coordinate pairs per second.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong widens _i32_ inputs to _u64_ outputs.
The `baseline_*` and `*_sqeuclideans3d` alternatives are the hand-written serial and broadcast loops of the benches.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `baseline_haversine`, `geo::Geodesic`, `ndarray_sqeuclideans3d`.

| NumKong                                                    |      Speed | Speedup | Alternative              |      Speed |
| :--------------------------------------------------------- | ---------: | ------: | :----------------------- | ---------: |
| `numkong::HaversinesPackedMatrix::haversines`, _f64 → f64_ |   304 MP/s |   12.8× | `geo::Haversine`         |  23.8 MP/s |
| `numkong::VincentysPackedMatrix::vincentys`, _f64 → f64_   |  24.2 MP/s |   7.53× | `baseline_vincenty`      |  3.21 MP/s |
| `numkong::SqEuclideans3D::sqeuclideans3d`, _i32 → u64_     | 4,465 MP/s |   2.33× | `baseline_sqeuclidean3d` | 1,920 MP/s |
| `numkong::SqEuclideans3D::sqeuclideans3d`, _f64 → f64_     | 1,308 MP/s |   0.49× | `baseline_sqeuclidean3d` | 2,651 MP/s |

> Measured October 14, 2026.

### Python

Also benchmarked: `sklearn.metrics.pairwise.haversine_distances`.

| NumKong                                  |      Speed | Speedup | Alternative                                 |     Speed |
| :--------------------------------------- | ---------: | ------: | :------------------------------------------ | --------: |
| `numkong.haversines_packed`, _f64 → f64_ |   292 MP/s |   8.58× | `numpy_haversines`                          | 34.1 MP/s |
| `numkong.vincentys_packed`, _f64 → f64_  |  24.4 MP/s |   19.7× | `pyproj.Geod.inv`                           | 1.24 MP/s |
| `numkong.sqeuclideans3d`, _i32 → u64_    | 4,474 MP/s |   5.42× | `scipy.spatial.distance.cdist`, _i32 → f64_ |  825 MP/s |
| `numkong.sqeuclideans3d`, _f64 → f64_    | 1,246 MP/s |   1.53× | `scipy.spatial.distance.cdist`              |  816 MP/s |

> Measured October 14, 2026.

## Run It

### Rust

```bash
# Default 1024×128 coordinate pairs
cargo bench --bench bench_geospatial --features bench_geospatial

# Square 512×512 coordinate pairs
NUMWARS_DIMS_HEIGHT=512 NUMWARS_DIMS_WIDTH=512 \
cargo bench --bench bench_geospatial --features bench_geospatial

# Focus on one metric
NUMWARS_FILTER="geospatial/haversines" \
cargo bench --bench bench_geospatial --features bench_geospatial
```

### Python

```bash
# Run the Python suite
uv run --with numkong,numpy,scipy,scikit-learn,pyproj,tabulate python geospatial/bench.py
```
