# Mesh Benchmarks

3D point-cloud alignment benchmarks comparing NumKong RMSD and Kabsch against scalar and nalgebra implementations.

## Rust

| Library                            | Precision     |        MP/s |
| :--------------------------------- | :------------ | ----------: |
| ___RMSD___                         |               |             |
| `numkong::MeshAlignment::rmsd`     | _f16 → f16_   | __2864.47__ |
| `numkong::MeshAlignment::rmsd`     | _bf16 → bf16_ | __2861.70__ |
| `numkong::MeshAlignment::rmsd`     | _f64 → f64_   | __1859.32__ |
| `numkong::MeshAlignment::rmsd`     | _f32 → f32_   | __1626.67__ |
| nalgebra-based RMSD                | _f32 → f32_   |     634.04 |
| ___Kabsch___                       |               |             |
| `numkong::MeshAlignment::kabsch`   | _f16 → f16_   |  __696.00__ |
| `numkong::MeshAlignment::kabsch`   | _bf16 → bf16_ |  __691.01__ |
| `numkong::MeshAlignment::kabsch`   | _f32 → f32_   |  __396.52__ |
| `numkong::MeshAlignment::kabsch`   | _f64 → f64_   |  __331.70__ |
| nalgebra-based Kabsch              | _f32 → f64_   |     283.16 |
| ___Umeyama___                      |               |             |
| `numkong::MeshAlignment::umeyama`  | _bf16 → bf16_ |  __673.50__ |
| `numkong::MeshAlignment::umeyama`  | _f16 → f16_   |  __614.06__ |
| `numkong::MeshAlignment::umeyama`  | _f32 → f32_   |  __376.48__ |
| `numkong::MeshAlignment::umeyama`  | _f64 → f64_   |  __325.16__ |
| nalgebra-based Umeyama             | _f32 → f64_   |     255.14 |

## Python

| Library                              | Precision   |        MP/s |
| :----------------------------------- | :---------- | ----------: |
| `numkong.rmsd`                       | _f64 → f64_ | __1311.77__ |
| `numkong.rmsd`                       | _f32 → f64_ | __1228.00__ |
| `numkong.kabsch`                     | _f32 → f64_ |  __360.08__ |
| `numkong.umeyama`                    | _f32 → f64_ |  __327.01__ |
| `numkong.umeyama`                    | _f64 → f64_ |  __296.67__ |
| `numkong.kabsch`                     | _f64 → f64_ |  __285.81__ |
| `numpy-based RMSD`                   | _f32 → f64_ |      124.48 |
| `numpy-based RMSD`                   | _f64 → f64_ |      117.30 |
| `biopython SVDSuperimposer (Kabsch)` | _f32 → f64_ |        2.88 |
| `biopython SVDSuperimposer (Kabsch)` | _f64 → f64_ |        2.92 |

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
