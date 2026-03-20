# Mesh Benchmarks

3D point-cloud alignment benchmarks comparing NumKong RMSD and Kabsch against scalar and nalgebra implementations.

## Rust

| Library                  | Precision     |       MP/s |
| :----------------------- | :------------ | ---------: |
| ___RMSD___               |               |            |
| `numkong::MeshAlignment` | _f32 → f32_   | __592.73__ |
| `numkong::MeshAlignment` | _f64 → f64_   | __971.35__ |
| `numkong::MeshAlignment` | _f16 → f16_   | __578.61__ |
| `numkong::MeshAlignment` | _bf16 → bf16_ | __567.69__ |
| serial code              | _f32 → f32_   |     532.85 |
| `nalgebra`               | _f32 → f32_   |     537.04 |
| ___Kabsch___             |               |            |
| `numkong::MeshAlignment` | _f32 → f32_   | __404.69__ |
| `numkong::MeshAlignment` | _f64 → f64_   | __245.90__ |
| `numkong::MeshAlignment` | _f16 → f16_   | __264.46__ |
| `numkong::MeshAlignment` | _bf16 → bf16_ | __272.09__ |
| serial code              | _f32 → f32_   |     116.84 |
| `nalgebra`               | _f32 → f32_   |     121.63 |
| ___Umeyama___            |               |            |
| `numkong::MeshAlignment` | _f32 → f32_   | __335.03__ |
| `numkong::MeshAlignment` | _f64 → f64_   | __147.75__ |
| `numkong::MeshAlignment` | _f16 → f16_   | __264.89__ |
| `numkong::MeshAlignment` | _bf16 → bf16_ | __268.63__ |
| serial code              | _f32 → f32_   |     111.35 |
| `nalgebra`               | _f32 → f32_   |     106.47 |

## Python

| Library                     | Precision   |       MP/s |
| :-------------------------- | :---------- | ---------: |
| ___RMSD___                  |             |            |
| `numkong.rmsd`              | _f32 → f64_ | __467.35__ |
| `numkong.rmsd`              | _f64 → f64_ | __825.51__ |
| `numpy`-based RMSD          | _f32 → f64_ |      50.49 |
| `numpy`-based RMSD          | _f64 → f64_ |      46.74 |
| ___Kabsch___                |             |            |
| `numkong.kabsch`            | _f32 → f64_ | __248.48__ |
| `numkong.kabsch`            | _f64 → f64_ | __238.79__ |
| `biopython.SVDSuperimposer` | _f32 → f64_ |       1.22 |
| `biopython.SVDSuperimposer` | _f64 → f64_ |       1.19 |
| ___Umeyama___               |             |            |
| `numkong.umeyama`           | _f32 → f64_ | __248.10__ |
| `numkong.umeyama`           | _f64 → f64_ | __159.25__ |

```bash
# Default 2048-point clouds
python mesh/bench.py

# Smaller 256-point clouds
python mesh/bench.py --count 256

# Focus on one operation
python mesh/bench.py -k "kabsch"
```

## Run It (Rust)

```bash
# Default 2048-point clouds
cargo bench --bench bench_mesh --features bench_mesh

# Smaller 256-point clouds
NUMWARS_DIMS=256 cargo bench --bench bench_mesh --features bench_mesh

# Focus on one operation
NUMWARS_FILTER="mesh/rmsd/f32" \
cargo bench --bench bench_mesh --features bench_mesh
```
