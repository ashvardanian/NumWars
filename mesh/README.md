# Mesh Benchmarks

3D point-cloud alignment benchmarks comparing NumKong RMSD and Kabsch against scalar and nalgebra implementations.

## Rust

| Library                  | Precision     |       MP/s |
| :----------------------- | :------------ | ---------: |
| ___RMSD___               |               |            |
| `numkong::MeshAlignment` | _f32 → f32_   | __610.32__ |
| `numkong::MeshAlignment` | _f64 → f64_   |          ? |
| `numkong::MeshAlignment` | _f16 → f16_   |          ? |
| `numkong::MeshAlignment` | _bf16 → bf16_ |          ? |
| serial code              | _f32 → f32_   |     214.54 |
| `nalgebra`               | _f32 → f32_   |     199.14 |
| ___Kabsch___             |               |            |
| `numkong::MeshAlignment` | _f32 → f32_   | __372.86__ |
| `numkong::MeshAlignment` | _f64 → f64_   |          ? |
| `numkong::MeshAlignment` | _f16 → f16_   |          ? |
| `numkong::MeshAlignment` | _bf16 → bf16_ |          ? |
| serial code              | _f32 → f32_   |     126.17 |
| `nalgebra`               | _f32 → f32_   |     125.14 |
| ___Umeyama___            |               |            |
| `numkong::MeshAlignment` | _f32 → f32_   |          ? |
| `numkong::MeshAlignment` | _f64 → f64_   |          ? |
| `numkong::MeshAlignment` | _f16 → f16_   |          ? |
| `numkong::MeshAlignment` | _bf16 → bf16_ |          ? |

## Python

| Library                        | Precision   |       MP/s |
| :----------------------------- | :---------- | ---------: |
| ___RMSD___                     |             |            |
| `numkong.rmsd`                 | _f32 → f64_ | __468.79__ |
| `numpy`-based RMSD             | _f32 → f64_ |      51.06 |
| ___Kabsch___                   |             |            |
| `numkong.kabsch`               | _f32 → f64_ | __260.75__ |
| `scipy.Rotation.align_vectors` | _f32 → f64_ |      13.51 |
| `biopython.SVDSuperimposer`    | _f32 → f64_ |       1.32 |
| ___Umeyama___                  |             |            |
| `numkong.umeyama`              | _f32 → f64_ | __245.37__ |

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
