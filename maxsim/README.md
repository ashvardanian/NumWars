# MaxSim Benchmarks

ColBERT-style late-interaction scoring benchmarks comparing NumKong against ndarray and NumPy.

Speeds are in scalar operations per second: GSO/s.
Speedup is NumKong's speed over the fastest alternative on the same chip, in the same language, on the same operation and input type.
A `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not measured on this chip.
NumKong widens its outputs: bf16 and f16 → f32, f32 → f64.

## Intel Xeon5 Emerald Rapids

### Rust

| NumKong                                            |       Speed | Speedup | Alternative                 |      Speed |
| :------------------------------------------------- | ----------: | ------: | :-------------------------- | ---------: |
| `numkong::MaxSimPackedMatrix::score`, _f32 → f64_  |  15.0 GSO/s |   0.16× | `ndarray::dot`, _f32 → f32_ | 91.8 GSO/s |
| `numkong::MaxSimPackedMatrix::score`, _bf16 → f32_ | 1,321 GSO/s |       – | –                           |          – |
| `numkong::MaxSimPackedMatrix::score`, _f16 → f32_  |  30.9 GSO/s |       – | –                           |          – |

> Measured October 14, 2026.

### Python

| NumKong                               |       Speed | Speedup | Alternative                 |     Speed |
| :------------------------------------ | ----------: | ------: | :-------------------------- | --------: |
| `numkong.maxsim_packed`, _f32 → f64_  |   110 GSO/s |   0.63× | `numpy.matmul`, _f32 → f32_ | 175 GSO/s |
| `numkong.maxsim_packed`, _bf16 → f32_ | 1,237 GSO/s |       – | –                           |         – |
| `numkong.maxsim_packed`, _f16 → f32_  |   194 GSO/s |       – | –                           |         – |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

| NumKong                                            |       Speed | Speedup | Alternative                 |      Speed |
| :------------------------------------------------- | ----------: | ------: | :-------------------------- | ---------: |
| `numkong::MaxSimPackedMatrix::score`, _f32 → f64_  | 1,483 GSO/s |   25.4× | `ndarray::dot`, _f32 → f32_ | 58.4 GSO/s |
| `numkong::MaxSimPackedMatrix::score`, _bf16 → f32_ |   984 GSO/s |       – | –                           |          – |
| `numkong::MaxSimPackedMatrix::score`, _f16 → f32_  |   980 GSO/s |       – | –                           |          – |

> Measured March 22, 2026.

### Python

| NumKong                               |       Speed | Speedup | Alternative                 |       Speed |
| :------------------------------------ | ----------: | ------: | :-------------------------- | ----------: |
| `numkong.maxsim_packed`, _f32 → f64_  | 2,426 GSO/s |   1.59× | `numpy.matmul`, _f32 → f32_ | 1,526 GSO/s |
| `numkong.maxsim_packed`, _bf16 → f32_ | 1,236 GSO/s |       – | –                           |           – |
| `numkong.maxsim_packed`, _f16 → f32_  |   697 GSO/s |       – | –                           |           – |

> Measured March 22, 2026.

## Run It

### Rust

```bash
# Default 1024×128×1536 workload
cargo bench --bench bench_maxsim --features bench_maxsim

# Smaller 128×128×256 workload
NUMWARS_DIMS_HEIGHT=128 NUMWARS_DIMS_WIDTH=128 NUMWARS_DIMS_DEPTH=256 \
cargo bench --bench bench_maxsim --features bench_maxsim

# Focus on one dtype
NUMWARS_FILTER="maxsim/f32" \
cargo bench --bench bench_maxsim --features bench_maxsim
```

### Python

```bash
uv run --with numkong,numpy,tabulate,ml_dtypes python maxsim/bench.py
```
