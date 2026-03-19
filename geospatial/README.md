# Geospatial Benchmarks

Batch geodesic distance benchmarks comparing NumKong against the geo and geopy libraries for Haversine and Vincenty formulas.

## Rust

| Library              | Precision   |       MP/s |
| :------------------- | :---------- | ---------: |
| ___Haversine___      |             |            |
| `numkong::haversine` | _f32 → f32_ | __564.53__ |
| `numkong::haversine` | _f64 → f64_ |          ? |
| `geo::GeoHaversine`  | _f64 → f64_ |      25.53 |
| ___Vincenty___       |             |            |
| `numkong::vincenty`  | _f32 → f32_ |  __57.76__ |
| `numkong::vincenty`  | _f64 → f64_ |          ? |
| `geo::Geodesic`      | _f64 → f64_ |       1.20 |

## Python

| Library              | Precision   |       MP/s |
| :------------------- | :---------- | ---------: |
| ___Haversine___      |             |            |
| `numkong.haversine`  | _f32 → f32_ | __526.05__ |
| `numkong.haversine`  | _f64 → f64_ |          ? |
| `geopy.great_circle` | _f32 → f64_ |       0.21 |
| ___Vincenty___       |             |            |
| `numkong.vincenty`   | _f32 → f32_ |  __57.22__ |
| `numkong.vincenty`   | _f64 → f64_ |          ? |
| `geopy.geodesic`     | _f32 → f64_ |       0.01 |

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
