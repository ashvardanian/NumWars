# Geospatial Benchmarks

Batch geodesic distance benchmarks comparing NumKong against the geo and geopy libraries for Haversine and Vincenty formulas.

## Rust

| Library              | Precision   |       MP/s |
| :------------------- | :---------- | ---------: |
| ___Haversine___      |             |            |
| `numkong::haversine` | _f32 → f32_ | __486.92__ |
| `numkong::haversine` | _f64 → f64_ | __151.65__ |
| `geo::GeoHaversine`  | _f32 → f32_ |      38.88 |
| `geo::GeoHaversine`  | _f64 → f64_ |      24.07 |
| ___Vincenty___       |             |            |
| `numkong::vincenty`  | _f32 → f32_ |  __68.96__ |
| `numkong::vincenty`  | _f64 → f64_ |  __17.79__ |
| `geo::Geodesic`      | _f64 → f64_ |       1.15 |

## Python

| Library              | Precision   |       MP/s |
| :------------------- | :---------- | ---------: |
| ___Haversine___      |             |            |
| `numkong.haversine`  | _f32 → f32_ | __475.41__ |
| `numkong.haversine`  | _f64 → f64_ | __154.92__ |
| `geopy.great_circle` | _f64 → f64_ |       0.18 |
| ___Vincenty___       |             |            |
| `numkong.vincenty`   | _f32 → f32_ |  __54.99__ |
| `numkong.vincenty`   | _f64 → f64_ |  __17.87__ |
| `geopy.geodesic`     | _f64 → f64_ |       0.01 |

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
