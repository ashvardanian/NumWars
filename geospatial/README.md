# Geospatial Benchmarks

Batch geodesic distance benchmarks comparing NumKong against the geo and geopy libraries for Haversine and Vincenty formulas.

## Rust

| Library                   | Precision   |       MP/s |
| :------------------------ | :---------- | ---------: |
| ___Haversine___           |             |            |
| `numkong::haversine`      | _f32 → f32_ | __491.98__ |
| `numkong::haversine`      | _f64 → f64_ | __149.72__ |
| serial baseline           | _f32 → f32_ |     137.83 |
| `geo::Haversine distance` | _f32 → f32_ |     136.96 |
| serial baseline           | _f64 → f64_ |      94.33 |
| `geo::Haversine distance` | _f64 → f64_ |      92.48 |
| ___Vincenty___            |             |            |
| `numkong::vincenty`       | _f32 → f32_ |  __71.64__ |
| serial baseline           | _f32 → f32_ |      18.20 |
| `numkong::vincenty`       | _f64 → f64_ |  __13.73__ |
| serial baseline           | _f64 → f64_ |       6.47 |
| `geo::Vincenty distance`  | _f64 → f64_ |       2.76 |

## Python

| Library                       | Precision   |       MP/s |
| :---------------------------- | :---------- | ---------: |
| `numkong.haversine`           | _f32 → f32_ | __444.38__ |
| `numkong.haversine`           | _f64 → f64_ | __132.85__ |
| `numkong.vincenty`            | _f32 → f32_ |  __65.89__ |
| `numkong.vincenty`            | _f64 → f64_ |  __11.93__ |
| `geopy.distance.great_circle` | _f64 → f64_ |       0.47 |
| `geopy.distance.geodesic`     | _f64 → f64_ |       0.03 |

## Run It

### Rust

```bash
# Default 2048 coordinate pairs
cargo bench --bench bench_geospatial --features bench_geospatial

# Smaller 256 coordinate pairs
NUMWARS_DIMS=256 \
cargo bench --bench bench_geospatial --features bench_geospatial

# Focus on one metric
NUMWARS_FILTER="geospatial/haversine/f32" \
cargo bench --bench bench_geospatial --features bench_geospatial
```

### Python

```bash
# Run the Python suite
uv run --with numkong,numpy,geopy,tabulate python geospatial/bench.py --count 2048
```
