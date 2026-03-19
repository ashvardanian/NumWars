# Mesh Benchmarks

3D point-cloud alignment benchmarks comparing NumKong RMSD and Kabsch against scalar and nalgebra implementations.

## Rust

| Library           | Precision |       MP/s |
| :---------------- | :-------: | ---------: |
| `numkong rmsd`    |   _f32_   | __610.32__ |
| `numkong kabsch`  |   _f32_   | __372.86__ |
| `scalar rmsd`     |   _f32_   |     214.54 |
| `nalgebra rmsd`   |   _f32_   |     199.14 |
| `scalar kabsch`   |   _f32_   |     126.17 |
| `nalgebra kabsch` |   _f32_   |     125.14 |

## Python

| Library                        | Operation |  Precision  |       MP/s |
| :----------------------------- | :-------: | :---------: | ---------: |
| `numkong.rmsd`                 |   RMSD    | _f32 → f64_ | __468.79__ |
| `numkong.kabsch`               |  Kabsch   | _f32 → f64_ | __260.75__ |
| `numkong.umeyama`              |  Umeyama  | _f32 → f64_ | __245.37__ |
| `numpy rmsd`                   |   RMSD    | _f32 → f64_ |      51.06 |
| `scipy Rotation.align_vectors` |  Kabsch   | _f32 → f64_ |      13.51 |
| `biopython SVDSuperimposer`    |  Kabsch   | _f32 → f64_ |       1.32 |

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
