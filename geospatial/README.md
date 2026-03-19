# Geospatial Benchmarks

Batch geodesic distance benchmarks comparing NumKong against the geo and geopy libraries for Haversine and Vincenty formulas.

## Rust

| Library             | Precision |       MP/s |
| :------------------ | :-------: | ---------: |
| `numkong haversine` |   `f32`   | **564.53** |
| `numkong vincenty`  |   `f32`   |  **57.76** |
| `geo haversine`     |   `f64`   |      25.53 |
| `geo vincenty`      |   `f64`   |       1.20 |

## Python

| Library              | Precision |       MP/s |
| :------------------- | :-------: | ---------: |
| `numkong haversine`  |   `f32`   | **526.05** |
| `numkong vincenty`   |   `f32`   |  **57.22** |
| `geopy great_circle` |   `f64`   |       0.21 |
| `geopy geodesic`     |   `f64`   |       0.01 |

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
