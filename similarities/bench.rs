//! Benchmark for many-to-many distance matrix operations
//!
//! Compares NumKong packed matrix operations vs BLAS-backed libraries
//! for computing N×M distance matrices.
//!
//! Competitors:
//! - numkong (DotsPackedMatrix SIMD, and the `*_symmetric` kernels for the SYRK groups)
//! - ndarray (BLAS-backed matmul for f32/f64)
//! - nalgebra (matmul-based for f32/f64)
//! - baseline (serial `count_ones` over 64-bit words for the SYRK Hamming and Jaccard groups)
//!
//! The SYRK groups score the batch A against itself, A × Aᵀ, where NumKong fills the upper
//! triangle and ndarray and nalgebra compute the full Gram matrix before the distance formula
//! visits its upper triangle. Their throughput counts only the n(n+1)/2 pairs of that triangle
//! times the depth, so a full product earns no credit for the mirrored half.
//!
//! A `cuda` build adds angular and Euclidean groups for each GPU in NUMWARS_DEVICES, named like
//! `similarities/angulars/bf16<cuda:0>`, for every dtype NumKong has CUDA kernels for, counting
//! pairs times the depth. Operands are random and live in unified memory, every round
//! synchronizes the stream, and the first round is checked against NumKong's CPU kernel on the
//! leading rows. The rival does what the CPU rivals do: a cuBLASLt product, then one epilogue
//! kernel applying norms computed beforehand.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_similarities --bench bench_similarities
//! NUMWARS_FILTER="angulars" cargo bench --features bench_similarities
//! NUMWARS_FILTER="similarities/symmetric" cargo bench --features bench_similarities
//! ```
//!
//! Benchmark naming: similarities/{metric}/{dtype} and similarities/symmetric/{metric}/{dtype}
//! Examples: similarities/angulars/f32, similarities/euclideans/f64, similarities/symmetric/hammings/u1x8

use std::hint::black_box;
use std::ops::Index;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use nalgebra::DMatrix;
use ndarray::Array2;
use num_traits::Float;
use numkong::prelude::*;
use numkong::{
    bf16, u1x8, Angulars, Capabilities, Euclideans, SymmetricAngularsOps, SymmetricEuclideansOps, SymmetricHammingsOps,
    SymmetricJaccardsOps,
};

use numwars::{pack_dots_matrix, propagate_thread_count, try_spawn_pool, Settings};

#[cfg(feature = "cuda")]
use std::ffi::{c_void, CString};

#[cfg(feature = "cuda")]
use cudarc::{cublaslt::sys::cudaDataType, driver::result as driver};
#[cfg(feature = "cuda")]
use numkong::{e2m1x2, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u4x2, NumberLike, StorageElement, UnifiedAllocator};
#[cfg(feature = "cuda")]
use numwars::{gpu_check, gpu_devices, CublasLtGemm, CublasLtLayout, GpuDevice, GpuMetric, GpuScalar};

// region: SYRK Baselines

/// Angular distances over the upper triangle of a Gram matrix, reading the norms off its diagonal.
fn angulars_from_gram<T: Float>(gram: &impl Index<(usize, usize), Output = T>, count: usize, output: &mut [T]) {
    for i in 0..count {
        for j in i..count {
            output[i * count + j] = T::one() - gram[(i, j)] / (gram[(i, i)] * gram[(j, j)]).sqrt();
        }
    }
}

/// Euclidean distances over the upper triangle of a Gram matrix, reading the norms off its diagonal.
fn euclideans_from_gram<T: Float>(gram: &impl Index<(usize, usize), Output = T>, count: usize, output: &mut [T]) {
    for i in 0..count {
        for j in i..count {
            let dot = gram[(i, j)];
            output[i * count + j] = (gram[(i, i)] + gram[(j, j)] - dot - dot).max(T::zero()).sqrt();
        }
    }
}

/// Copies `count` rows of bits into zero-padded 64-bit words, the layout the baselines below scan.
fn bit_words(vectors: &[u1x8], count: usize) -> Vec<u64> {
    let bytes_per_row = vectors.len() / count;
    let words_per_row = bytes_per_row.div_ceil(8);
    let mut words = vec![0u64; count * words_per_row];
    for (row, row_words) in vectors.chunks(bytes_per_row).zip(words.chunks_mut(words_per_row)) {
        for (byte_index, byte) in row.iter().enumerate() {
            row_words[byte_index / 8] |= u64::from(byte.bits()) << (byte_index % 8 * 8);
        }
    }
    words
}

/// Hamming distances between `count` rows of `words`, upper triangle only.
fn baseline_hammings_symmetric(words: &[u64], count: usize, output: &mut [u32]) {
    let words_per_row = words.len() / count;
    for (i, first) in words.chunks(words_per_row).enumerate() {
        for (j, second) in words.chunks(words_per_row).enumerate().skip(i) {
            output[i * count + j] = first.iter().zip(second).map(|(a, b)| (a ^ b).count_ones()).sum();
        }
    }
}

/// Jaccard distances between `count` rows of `words`, upper triangle only.
fn baseline_jaccards_symmetric(words: &[u64], count: usize, output: &mut [f32]) {
    let words_per_row = words.len() / count;
    for (i, first) in words.chunks(words_per_row).enumerate() {
        for (j, second) in words.chunks(words_per_row).enumerate().skip(i) {
            let (intersection, union) = first
                .iter()
                .zip(second)
                .fold((0u32, 0u32), |(intersection, union), (a, b)| {
                    (intersection + (a & b).count_ones(), union + (a | b).count_ones())
                });
            output[i * count + j] = if union == 0 {
                0.0
            } else {
                1.0 - intersection as f32 / union as f32
            };
        }
    }
}

// endregion

// region: Per-Library Run Traits

trait RunNumKongAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize, _threads: usize) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _bs: usize, _dim: usize, _threads: usize) {}
}

impl<T: Angulars + Clone + Send + Sync + 'static> RunNumKongAngulars for T
where
    T::Accumulator: Send + Sync,
    <T as Angulars>::SpatialResult: Send + Sync,
{
    fn run(
        group: &mut BenchmarkGroup<'_, WallTime>,
        data_a: &[T],
        data_b: &[T],
        bs: usize,
        dim: usize,
        threads: usize,
    ) {
        let tensor_a = Tensor::<T>::from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let tensor_b = Tensor::<T>::from_slice(data_b, &[bs, dim]).expect("Failed to create tensor B");
        match try_spawn_pool(threads) {
            Some(mut pool) => {
                let packed_b = pack_dots_matrix(&tensor_b, Some(&mut pool));
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.angulars_packed_parallel(&packed_b, &mut pool)))
                });
            }
            None => {
                let packed_b = pack_dots_matrix(&tensor_b, None);
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.angulars_packed(&packed_b)))
                });
            }
        }
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], bs: usize, dim: usize, threads: usize) {
        let tensor_a = Tensor::<T>::from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let mut output = Tensor::<T::SpatialResult>::zeros(&[bs, bs]).expect("Failed to allocate the output");
        match try_spawn_pool(threads) {
            Some(mut pool) => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .angulars_symmetric_parallel_into(&mut output, &mut pool)
                            .expect("parallel symmetric angulars failed")
                    })
                });
            }
            None => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .angulars_symmetric_into(&mut output)
                            .expect("symmetric angulars failed")
                    })
                });
            }
        }
    }
}

trait RunNdarrayAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNdarrayAngulars for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = Array2::<f32>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f32>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a: Vec<f32> = matrix_a.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let norms_b: Vec<f32> = matrix_b.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let mut output = Array2::<f32>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], bs: usize, dim: usize) {
        let matrix_a = Array2::<f32>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let mut output = vec![0.0f32; bs * bs];

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                angulars_from_gram(&matrix_a.dot(&matrix_a.t()), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayAngulars for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = Array2::<f64>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f64>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a: Vec<f64> = matrix_a.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let norms_b: Vec<f64> = matrix_b.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        let mut output = Array2::<f64>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], bs: usize, dim: usize) {
        let matrix_a = Array2::<f64>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let mut output = vec![0.0f64; bs * bs];

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                angulars_from_gram(&matrix_a.dot(&matrix_a.t()), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayAngulars for i8 {}
impl RunNdarrayAngulars for u8 {}
impl RunNdarrayAngulars for bf16 {}

trait RunNalgebraAngulars: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNalgebraAngulars for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f32>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f32>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a: Vec<f32> = (0..bs).map(|i| matrix_a.row(i).norm()).collect();
        let norms_b: Vec<f32> = (0..bs).map(|i| matrix_b.row(i).norm()).collect();
        let mut output = DMatrix::<f32>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f32>::from_row_slice(bs, dim, data_a);
        let matrix_a_t = matrix_a.transpose();
        let mut output = vec![0.0f32; bs * bs];

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                angulars_from_gram(&(&matrix_a * &matrix_a_t), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraAngulars for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f64>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f64>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a: Vec<f64> = (0..bs).map(|i| matrix_a.row(i).norm()).collect();
        let norms_b: Vec<f64> = (0..bs).map(|i| matrix_b.row(i).norm()).collect();
        let mut output = DMatrix::<f64>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        output[(i, j)] = 1.0 - scores[(i, j)] / (norms_a[i] * norms_b[j]);
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f64>::from_row_slice(bs, dim, data_a);
        let matrix_a_t = matrix_a.transpose();
        let mut output = vec![0.0f64; bs * bs];

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                angulars_from_gram(&(&matrix_a * &matrix_a_t), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraAngulars for i8 {}
impl RunNalgebraAngulars for u8 {}
impl RunNalgebraAngulars for bf16 {}

trait RunNumKongEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize, _threads: usize) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _bs: usize, _dim: usize, _threads: usize) {}
}

impl<T: Euclideans + Clone + Send + Sync + 'static> RunNumKongEuclideans for T
where
    T::Accumulator: Send + Sync,
    <T as Euclideans>::SpatialResult: Send + Sync,
{
    fn run(
        group: &mut BenchmarkGroup<'_, WallTime>,
        data_a: &[T],
        data_b: &[T],
        bs: usize,
        dim: usize,
        threads: usize,
    ) {
        let tensor_a = Tensor::<T>::from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let tensor_b = Tensor::<T>::from_slice(data_b, &[bs, dim]).expect("Failed to create tensor B");
        match try_spawn_pool(threads) {
            Some(mut pool) => {
                let packed_b = pack_dots_matrix(&tensor_b, Some(&mut pool));
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.euclideans_packed_parallel(&packed_b, &mut pool)))
                });
            }
            None => {
                let packed_b = pack_dots_matrix(&tensor_b, None);
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.euclideans_packed(&packed_b)))
                });
            }
        }
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[T], bs: usize, dim: usize, threads: usize) {
        let tensor_a = Tensor::<T>::from_slice(data_a, &[bs, dim]).expect("Failed to create tensor A");
        let mut output = Tensor::<T::SpatialResult>::zeros(&[bs, bs]).expect("Failed to allocate the output");
        match try_spawn_pool(threads) {
            Some(mut pool) => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .euclideans_symmetric_parallel_into(&mut output, &mut pool)
                            .expect("parallel symmetric euclideans failed")
                    })
                });
            }
            None => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .euclideans_symmetric_into(&mut output)
                            .expect("symmetric euclideans failed")
                    })
                });
            }
        }
    }
}

trait RunNdarrayEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNdarrayEuclideans for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = Array2::<f32>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f32>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a_sq: Vec<f32> = matrix_a.rows().into_iter().map(|row| row.dot(&row)).collect();
        let norms_b_sq: Vec<f32> = matrix_b.rows().into_iter().map(|row| row.dot(&row)).collect();
        let mut output = Array2::<f32>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], bs: usize, dim: usize) {
        let matrix_a = Array2::<f32>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let mut output = vec![0.0f32; bs * bs];

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                euclideans_from_gram(&matrix_a.dot(&matrix_a.t()), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayEuclideans for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = Array2::<f64>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let matrix_b = Array2::<f64>::from_shape_vec((bs, dim), data_b.to_vec()).expect("Failed to create B");
        let norms_a_sq: Vec<f64> = matrix_a.rows().into_iter().map(|row| row.dot(&row)).collect();
        let norms_b_sq: Vec<f64> = matrix_b.rows().into_iter().map(|row| row.dot(&row)).collect();
        let mut output = Array2::<f64>::zeros((bs, bs));

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let scores = matrix_a.dot(&matrix_b.t());
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], bs: usize, dim: usize) {
        let matrix_a = Array2::<f64>::from_shape_vec((bs, dim), data_a.to_vec()).expect("Failed to create A");
        let mut output = vec![0.0f64; bs * bs];

        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                euclideans_from_gram(&matrix_a.dot(&matrix_a.t()), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNdarrayEuclideans for i8 {}
impl RunNdarrayEuclideans for u8 {}
impl RunNdarrayEuclideans for bf16 {}

trait RunNalgebraEuclideans: Sized {
    fn run(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _b: &[Self], _bs: usize, _dim: usize) {}
    fn run_symmetric(_g: &mut BenchmarkGroup<'_, WallTime>, _a: &[Self], _bs: usize, _dim: usize) {}
}

impl RunNalgebraEuclideans for f32 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], data_b: &[f32], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f32>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f32>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a_sq: Vec<f32> = (0..bs).map(|i| matrix_a.row(i).dot(&matrix_a.row(i))).collect();
        let norms_b_sq: Vec<f32> = (0..bs).map(|i| matrix_b.row(i).dot(&matrix_b.row(i))).collect();
        let mut output = DMatrix::<f32>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f32], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f32>::from_row_slice(bs, dim, data_a);
        let matrix_a_t = matrix_a.transpose();
        let mut output = vec![0.0f32; bs * bs];

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                euclideans_from_gram(&(&matrix_a * &matrix_a_t), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraEuclideans for f64 {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], data_b: &[f64], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f64>::from_row_slice(bs, dim, data_a);
        let matrix_b = DMatrix::<f64>::from_row_slice(bs, dim, data_b);
        let matrix_b_t = matrix_b.transpose();
        let norms_a_sq: Vec<f64> = (0..bs).map(|i| matrix_a.row(i).dot(&matrix_a.row(i))).collect();
        let norms_b_sq: Vec<f64> = (0..bs).map(|i| matrix_b.row(i).dot(&matrix_b.row(i))).collect();
        let mut output = DMatrix::<f64>::zeros(bs, bs);

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                let scores = &matrix_a * &matrix_b_t;
                for i in 0..bs {
                    for j in 0..bs {
                        let sq = (norms_a_sq[i] + norms_b_sq[j] - 2.0 * scores[(i, j)]).max(0.0);
                        output[(i, j)] = sq.sqrt();
                    }
                }
                black_box(&output);
            })
        });
    }

    fn run_symmetric(group: &mut BenchmarkGroup<'_, WallTime>, data_a: &[f64], bs: usize, dim: usize) {
        let matrix_a = DMatrix::<f64>::from_row_slice(bs, dim, data_a);
        let matrix_a_t = matrix_a.transpose();
        let mut output = vec![0.0f64; bs * bs];

        group.bench_function("nalgebra", |bench| {
            bench.iter(|| {
                euclideans_from_gram(&(&matrix_a * &matrix_a_t), bs, &mut output);
                black_box(&output);
            })
        });
    }
}

impl RunNalgebraEuclideans for i8 {}
impl RunNalgebraEuclideans for u8 {}
impl RunNalgebraEuclideans for bf16 {}

// endregion

// region: Generic Helpers

/// Work a SYRK group is credited with: the n(n+1)/2 pairs of one triangle times the depth.
fn symmetric_throughput(count: usize, depth: usize) -> Throughput {
    Throughput::Elements((count * (count + 1) / 2 * depth) as u64)
}

fn bench_angulars_dtype<T>(c: &mut Criterion, settings: &Settings, dtype: &str, init: T)
where
    T: Clone + RunNumKongAngulars + RunNdarrayAngulars + RunNalgebraAngulars + 'static,
{
    let (batch_size, dimension) = (settings.batch_size(), settings.dims);
    let total_elements = batch_size * dimension;
    let data_a = vec![init.clone(); total_elements];

    let name = format!("similarities/angulars/{dtype}");
    if settings.selects(&name) {
        let mut group = c.benchmark_group(name);
        let data_b = vec![init; total_elements];
        group.throughput(Throughput::Bytes(
            (2 * total_elements * std::mem::size_of::<T>()) as u64,
        ));

        <T as RunNumKongAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension, settings.threads);
        <T as RunNdarrayAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
        <T as RunNalgebraAngulars>::run(&mut group, &data_a, &data_b, batch_size, dimension);
        group.finish();
    }

    let name = format!("similarities/symmetric/angulars/{dtype}");
    if settings.selects(&name) {
        let mut group = c.benchmark_group(name);
        group.throughput(symmetric_throughput(batch_size, dimension));

        <T as RunNumKongAngulars>::run_symmetric(&mut group, &data_a, batch_size, dimension, settings.threads);
        <T as RunNdarrayAngulars>::run_symmetric(&mut group, &data_a, batch_size, dimension);
        <T as RunNalgebraAngulars>::run_symmetric(&mut group, &data_a, batch_size, dimension);
        group.finish();
    }
}

fn bench_euclideans_dtype<T>(c: &mut Criterion, settings: &Settings, dtype: &str, init: T)
where
    T: Clone + RunNumKongEuclideans + RunNdarrayEuclideans + RunNalgebraEuclideans + 'static,
{
    let (batch_size, dimension) = (settings.batch_size(), settings.dims);
    let total_elements = batch_size * dimension;
    let data_a = vec![init.clone(); total_elements];

    let name = format!("similarities/euclideans/{dtype}");
    if settings.selects(&name) {
        let mut group = c.benchmark_group(name);
        let data_b = vec![init; total_elements];
        group.throughput(Throughput::Bytes(
            (2 * total_elements * std::mem::size_of::<T>()) as u64,
        ));

        <T as RunNumKongEuclideans>::run(&mut group, &data_a, &data_b, batch_size, dimension, settings.threads);
        <T as RunNdarrayEuclideans>::run(&mut group, &data_a, &data_b, batch_size, dimension);
        <T as RunNalgebraEuclideans>::run(&mut group, &data_a, &data_b, batch_size, dimension);
        group.finish();
    }

    let name = format!("similarities/symmetric/euclideans/{dtype}");
    if settings.selects(&name) {
        let mut group = c.benchmark_group(name);
        group.throughput(symmetric_throughput(batch_size, dimension));

        <T as RunNumKongEuclideans>::run_symmetric(&mut group, &data_a, batch_size, dimension, settings.threads);
        <T as RunNdarrayEuclideans>::run_symmetric(&mut group, &data_a, batch_size, dimension);
        <T as RunNalgebraEuclideans>::run_symmetric(&mut group, &data_a, batch_size, dimension);
        group.finish();
    }
}

// endregion

// region: Benchmarks

/// Benchmark N×M angular distance matrix.
pub fn bench_angulars(c: &mut Criterion, settings: &Settings) {
    bench_angulars_dtype(c, settings, "f32", 1.0f32);
    bench_angulars_dtype(c, settings, "f64", 1.0f64);
    bench_angulars_dtype(c, settings, "i8", 1i8);
    bench_angulars_dtype(c, settings, "u8", 1u8);
    bench_angulars_dtype(c, settings, "bf16", bf16::from_f32(1.0));
}

/// Benchmark N×M Euclidean distance matrix.
pub fn bench_euclideans(c: &mut Criterion, settings: &Settings) {
    bench_euclideans_dtype(c, settings, "f32", 1.0f32);
    bench_euclideans_dtype(c, settings, "f64", 1.0f64);
    bench_euclideans_dtype(c, settings, "i8", 1i8);
    bench_euclideans_dtype(c, settings, "u8", 1u8);
    bench_euclideans_dtype(c, settings, "bf16", bf16::from_f32(1.0));
}

/// Benchmark N×M Hamming distance matrix.
pub fn bench_hammings(c: &mut Criterion, settings: &Settings) {
    let dimension = settings.dims;
    let byte_count = dimension.div_ceil(8);
    let batch_size = settings.batch_size();

    if settings.selects("similarities/hammings/u1x8") {
        let mut group = c.benchmark_group("similarities/hammings/u1x8");
        let total_bytes = batch_size * byte_count;
        let matrix_a_data = vec![u1x8::new(0xAA); total_bytes];
        let matrix_b_data = vec![u1x8::new(0x55); total_bytes];
        group.throughput(Throughput::Bytes((total_bytes * 2) as u64));

        let tensor_a =
            Tensor::<u1x8>::from_slice(&matrix_a_data, &[batch_size, dimension]).expect("Failed to create tensor A");
        let tensor_b =
            Tensor::<u1x8>::from_slice(&matrix_b_data, &[batch_size, dimension]).expect("Failed to create tensor B");

        match try_spawn_pool(settings.threads) {
            Some(mut pool) => {
                let packed_b = pack_dots_matrix(&tensor_b, Some(&mut pool));
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.hammings_packed_parallel(&packed_b, &mut pool)))
                });
            }
            None => {
                let packed_b = pack_dots_matrix(&tensor_b, None);
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.hammings_packed(&packed_b)))
                });
            }
        }
        group.finish();
    }

    if settings.selects("similarities/symmetric/hammings/u1x8") {
        let mut group = c.benchmark_group("similarities/symmetric/hammings/u1x8");
        let matrix_a_data = vec![u1x8::new(0xAA); batch_size * byte_count];
        group.throughput(symmetric_throughput(batch_size, dimension));

        let tensor_a =
            Tensor::<u1x8>::from_slice(&matrix_a_data, &[batch_size, dimension]).expect("Failed to create tensor A");
        let mut output = Tensor::<u32>::zeros(&[batch_size, batch_size]).expect("Failed to allocate the output");
        match try_spawn_pool(settings.threads) {
            Some(mut pool) => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .hammings_symmetric_parallel_into(&mut output, &mut pool)
                            .expect("parallel symmetric hammings failed")
                    })
                });
            }
            None => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .hammings_symmetric_into(&mut output)
                            .expect("symmetric hammings failed")
                    })
                });
            }
        }

        let words = bit_words(&matrix_a_data, batch_size);
        let mut baseline_output = vec![0u32; batch_size * batch_size];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_hammings_symmetric(&words, batch_size, &mut baseline_output);
                black_box(&baseline_output);
            })
        });
        group.finish();
    }
}

/// Benchmark N×M Jaccard distance matrix.
pub fn bench_jaccards(c: &mut Criterion, settings: &Settings) {
    let dimension = settings.dims;
    let byte_count = dimension.div_ceil(8);
    let batch_size = settings.batch_size();

    if settings.selects("similarities/jaccards/u1x8") {
        let mut group = c.benchmark_group("similarities/jaccards/u1x8");
        let total_bytes = batch_size * byte_count;
        let matrix_a_data = vec![u1x8::new(0xAA); total_bytes];
        let matrix_b_data = vec![u1x8::new(0x55); total_bytes];
        group.throughput(Throughput::Bytes((total_bytes * 2) as u64));

        let tensor_a =
            Tensor::<u1x8>::from_slice(&matrix_a_data, &[batch_size, dimension]).expect("Failed to create tensor A");
        let tensor_b =
            Tensor::<u1x8>::from_slice(&matrix_b_data, &[batch_size, dimension]).expect("Failed to create tensor B");

        match try_spawn_pool(settings.threads) {
            Some(mut pool) => {
                let packed_b = pack_dots_matrix(&tensor_b, Some(&mut pool));
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.jaccards_packed_parallel(&packed_b, &mut pool)))
                });
            }
            None => {
                let packed_b = pack_dots_matrix(&tensor_b, None);
                group.bench_function("numkong", |bench| {
                    bench.iter(|| black_box(tensor_a.jaccards_packed(&packed_b)))
                });
            }
        }
        group.finish();
    }

    if settings.selects("similarities/symmetric/jaccards/u1x8") {
        let mut group = c.benchmark_group("similarities/symmetric/jaccards/u1x8");
        let matrix_a_data = vec![u1x8::new(0xAA); batch_size * byte_count];
        group.throughput(symmetric_throughput(batch_size, dimension));

        let tensor_a =
            Tensor::<u1x8>::from_slice(&matrix_a_data, &[batch_size, dimension]).expect("Failed to create tensor A");
        let mut output = Tensor::<f32>::zeros(&[batch_size, batch_size]).expect("Failed to allocate the output");
        match try_spawn_pool(settings.threads) {
            Some(mut pool) => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .jaccards_symmetric_parallel_into(&mut output, &mut pool)
                            .expect("parallel symmetric jaccards failed")
                    })
                });
            }
            None => {
                group.bench_function("numkong", |bench| {
                    bench.iter(|| {
                        tensor_a
                            .jaccards_symmetric_into(&mut output)
                            .expect("symmetric jaccards failed")
                    })
                });
            }
        }

        let words = bit_words(&matrix_a_data, batch_size);
        let mut baseline_output = vec![0.0f32; batch_size * batch_size];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                baseline_jaccards_symmetric(&words, batch_size, &mut baseline_output);
                black_box(&baseline_output);
            })
        });
        group.finish();
    }
}

// endregion

// region: GPU Rows

/// Leading rows of a GPU result its first round is checked on.
#[cfg(feature = "cuda")]
const CHECKED_ROWS: usize = 16;

/// Relative error a GPU row may show against the CPU, which sums the depth in another order.
#[cfg(feature = "cuda")]
const GPU_TOLERANCE: f64 = 1e-3;

/// The rival's epilogue, turning its cuBLASLt products into distances in place from norms computed
/// beforehand: inverse norms for angular distances, squared ones for Euclidean.
#[cfg(feature = "cuda")]
const EPILOGUE: &str = r#"
extern "C" __global__ void epilogue(PRODUCT const *products, double const *row_norms, double const *column_norms,
                                    REAL *distances, unsigned long long rows, unsigned long long columns) {
    unsigned long long index = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (index >= rows * columns) return;
    double product = products[index], row = row_norms[index / columns], column = column_norms[index % columns];
#if ANGULAR
    distances[index] = (REAL)(1 - product * row * column);
#else
    distances[index] = (REAL)sqrt(fmax(row + column - 2 * product, 0.0));
#endif
}
"#;

/// A rival added to a GPU group, given the metric, both operands and the CPU's leading rows.
#[cfg(feature = "cuda")]
type Rival<T> = fn(
    &mut BenchmarkGroup<'_, WallTime>,
    &str,
    &GpuDevice,
    GpuMetric,
    &Tensor<T, UnifiedAllocator>,
    &Tensor<T, UnifiedAllocator>,
    &[f64],
    bool,
);

/// Adds the cuBLASLt product and its epilogue to `group` as `cublaslt`, once its first round matches
/// `expected`; a SYRK group's rival computes the full product, as the CPU rivals do.
#[cfg(feature = "cuda")]
fn bench_cublaslt<T: GpuScalar + NumberLike>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    device: &GpuDevice,
    metric: GpuMetric,
    a: &Tensor<T, UnifiedAllocator>,
    b: &Tensor<T, UnifiedAllocator>,
    expected: &[f64],
    upper_only: bool,
) {
    let Some(types) = T::CUBLASLT else {
        return;
    };
    let (rows, columns, depth) = (a.shape()[0], b.shape()[0], a.shape()[1]);
    let gemm = match CublasLtGemm::new(device, types, CublasLtLayout::new(rows, columns, depth)) {
        Ok(gemm) => gemm,
        Err(error) => return println!("{name}/cublaslt is unavailable: {error}"),
    };
    let wide = types.output == cudaDataType::CUDA_R_64F;
    let product = match types.output {
        cudaDataType::CUDA_R_64F => "double",
        cudaDataType::CUDA_R_32I => "int",
        _ => "float",
    };
    let angular = matches!(metric, GpuMetric::Angulars) as u8;
    let real = if wide { "double" } else { "float" };
    let source = format!("#define PRODUCT {product}\n#define REAL {real}\n#define ANGULAR {angular}\n{EPILOGUE}");
    let ptx = cudarc::nvrtc::compile_ptx(source).expect("NVRTC failed to compile the epilogue");
    let image = CString::new(ptx.to_src()).expect("PTX holds no zero bytes");
    let module = unsafe { driver::module::load_data(image.as_ptr().cast()) }.expect("Failed to load the epilogue");
    let entry = CString::new("epilogue").expect("A plain name");
    let function = unsafe { driver::module::get_function(module, entry) }.expect("No epilogue in the module");
    let norms = |matrix: &Tensor<T, UnifiedAllocator>| {
        let squares = matrix
            .as_slice()
            .chunks(depth)
            .map(|row| row.iter().map(|value| value.to_f64().powi(2)).sum::<f64>());
        let norms: Vec<f64> = match metric {
            GpuMetric::Angulars => squares.map(|square: f64| 1.0 / square.sqrt()).collect(),
            _ => squares.collect(),
        };
        Tensor::<f64, UnifiedAllocator>::from_slice_in(&norms, &[norms.len()], device.allocator())
            .expect("Failed to allocate the norms")
    };
    let (row_norms, column_norms) = (norms(a), norms(b));
    let mut products =
        Tensor::<T::Product, _>::zeros_in(&[rows, columns], device.allocator()).expect("Failed to allocate");
    let launch = |products: &mut Tensor<T::Product, UnifiedAllocator>| {
        gemm.launch(a, b, products);
        let (target, row_pointer, column_pointer) = (products.as_mut_ptr(), row_norms.as_ptr(), column_norms.as_ptr());
        let (rows, columns) = (rows as u64, columns as u64);
        let mut arguments = [
            (&target as *const *mut T::Product).cast_mut().cast::<c_void>(),
            (&row_pointer as *const *const f64).cast_mut().cast(),
            (&column_pointer as *const *const f64).cast_mut().cast(),
            (&target as *const *mut T::Product).cast_mut().cast(),
            (&rows as *const u64).cast_mut().cast(),
            (&columns as *const u64).cast_mut().cast(),
        ];
        let blocks = (rows * columns).div_ceil(256) as u32;
        unsafe {
            driver::launch_kernel(
                function,
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                device.stream.cast(),
                &mut arguments,
            )
        }
        .expect("Failed to launch the epilogue");
        device.synchronize();
    };
    launch(&mut products);
    let checked = rows.min(CHECKED_ROWS) * columns;
    let found: Vec<f64> = match wide {
        true => unsafe { std::slice::from_raw_parts(products.as_ptr().cast::<f64>(), checked) }.to_vec(),
        false => unsafe { std::slice::from_raw_parts(products.as_ptr().cast::<f32>(), checked) }
            .iter()
            .map(|&value| value.into())
            .collect(),
    };
    match gpu_check(
        &format!("{name}/cublaslt"),
        &found,
        expected,
        columns,
        upper_only,
        GPU_TOLERANCE,
    ) {
        Ok(()) => {
            group.bench_function("cublaslt", |bench| bench.iter(|| launch(&mut products)));
        }
        Err(error) => println!("Skipping a rival that {error}"),
    }
    unsafe { driver::module::unload(module) }.expect("Failed to unload the epilogue");
}

#[cfg(feature = "cuda")]
fn cpu_angulars<T: Angulars>(rows: &Tensor<T>, packed: &DotsPackedMatrix<T>) -> Tensor<T::SpatialResult> {
    rows.angulars_packed(packed).expect("CPU angulars failed")
}

#[cfg(feature = "cuda")]
fn cpu_euclideans<T: Euclideans>(rows: &Tensor<T>, packed: &DotsPackedMatrix<T>) -> Tensor<T::SpatialResult> {
    rows.euclideans_packed(packed).expect("CPU euclideans failed")
}

/// GEMM and SYRK groups of `metric` over `T` on each GPU, over random operands in unified memory,
/// each round synchronized and each first round checked against NumKong's CPU kernel.
#[cfg(feature = "cuda")]
fn bench_metric_gpu<T: GpuScalar, Output: StorageElement + Default + Into<f64>>(
    c: &mut Criterion,
    settings: &Settings,
    devices: &[GpuDevice],
    (metric, family): (GpuMetric, &str),
    dtype: &str,
    reference: fn(&Tensor<T>, &DotsPackedMatrix<T>) -> Tensor<Output>,
    rival: Option<Rival<T>>,
) {
    let (count, depth) = (settings.batch_size(), settings.dims);
    let seed = u64::from(settings.seed);
    for device in devices {
        for symmetric in [false, true] {
            let kernel = if symmetric { "symmetric/" } else { "" };
            let name = format!("similarities/{kernel}{family}/{dtype}{}", device.label());
            if !settings.selects(&name) {
                continue;
            }
            let a = device.random::<T>(&[count, depth], seed);
            let b = (!symmetric).then(|| device.random::<T>(&[count, depth], seed + 1));
            let b = b.as_ref().unwrap_or(&a);
            let packed = match symmetric {
                true => None,
                false => device.pack(b),
            };
            let mut distances =
                Tensor::<Output, _>::zeros_in(&[count, count], device.allocator()).expect("Failed to allocate");
            let launch = |distances: &mut Tensor<Output, UnifiedAllocator>| -> Option<()> {
                match &packed {
                    Some(packed) => device.packed(metric, &a, packed, distances),
                    None => device.symmetric(metric, &a, distances),
                }?;
                device.synchronize();
                Some(())
            };
            if (!symmetric && packed.is_none()) || launch(&mut distances).is_none() {
                println!("{name} is unavailable: NumKong has no CUDA kernel for {dtype}");
                continue;
            }
            let leading = count.min(CHECKED_ROWS);
            let values = leading * depth / T::dimensions_per_value();
            let rows =
                Tensor::<T>::from_slice(&a.as_slice()[..values], &[leading, depth]).expect("Failed to copy rows");
            let expected = reference(&rows, &DotsPackedMatrix::new(b).expect("Failed to pack on the CPU"));
            let expected: Vec<f64> = expected.as_slice().iter().map(|&value| value.into()).collect();
            gpu_check(&name, distances.as_slice(), &expected, count, symmetric, GPU_TOLERANCE)
                .unwrap_or_else(|error| panic!("{error}"));
            let mut group = c.benchmark_group(&name);
            let pairs = if symmetric {
                count * (count + 1) / 2
            } else {
                count * count
            };
            group.throughput(Throughput::Elements((pairs * depth) as u64));
            group.bench_function("numkong", |bench| bench.iter(|| launch(&mut distances)));
            if let Some(rival) = rival {
                rival(&mut group, &name, device, metric, &a, b, &expected, symmetric);
            }
            group.finish();
        }
    }
}

/// Both metrics of `T` on each GPU.
#[cfg(feature = "cuda")]
fn bench_dtype_gpu<T, Output>(
    c: &mut Criterion,
    settings: &Settings,
    devices: &[GpuDevice],
    dtype: &str,
    rival: Option<Rival<T>>,
) where
    T: GpuScalar + Angulars<SpatialResult = Output> + Euclideans<SpatialResult = Output>,
    Output: StorageElement + Default + Into<f64>,
{
    let angulars = (GpuMetric::Angulars, "angulars");
    let euclideans = (GpuMetric::Euclideans, "euclideans");
    bench_metric_gpu(c, settings, devices, angulars, dtype, cpu_angulars::<T>, rival);
    bench_metric_gpu(c, settings, devices, euclideans, dtype, cpu_euclideans::<T>, rival);
}

/// Every dtype NumKong's CUDA kernels take, on each GPU; cuBLASLt races the ones it multiplies.
#[cfg(feature = "cuda")]
fn bench_similarities_gpu(c: &mut Criterion, settings: &Settings, devices: &[GpuDevice]) {
    bench_dtype_gpu::<f32, _>(c, settings, devices, "f32", Some(bench_cublaslt::<f32>));
    bench_dtype_gpu::<f64, _>(c, settings, devices, "f64", Some(bench_cublaslt::<f64>));
    bench_dtype_gpu::<i8, _>(c, settings, devices, "i8", Some(bench_cublaslt::<i8>));
    bench_dtype_gpu::<u8, _>(c, settings, devices, "u8", None);
    bench_dtype_gpu::<bf16, _>(c, settings, devices, "bf16", Some(bench_cublaslt::<bf16>));
    bench_dtype_gpu::<f16, _>(c, settings, devices, "f16", Some(bench_cublaslt::<f16>));
    bench_dtype_gpu::<e4m3, _>(c, settings, devices, "e4m3", Some(bench_cublaslt::<e4m3>));
    bench_dtype_gpu::<e5m2, _>(c, settings, devices, "e5m2", None);
    bench_dtype_gpu::<e2m3, _>(c, settings, devices, "e2m3", None);
    bench_dtype_gpu::<e3m2, _>(c, settings, devices, "e3m2", None);
    bench_dtype_gpu::<e2m1x2, _>(c, settings, devices, "e2m1", None);
    bench_dtype_gpu::<i4x2, _>(c, settings, devices, "i4", None);
    bench_dtype_gpu::<u4x2, _>(c, settings, devices, "u4", None);
}

// endregion

// region: Tests

// Without `--test`, the `harness = false` build strips every `#[test]` and leaves these helpers unused
#[cfg(test)]
#[allow(dead_code)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    /// Small integers keep every product and sum exact in `f32`.
    fn small_vectors(count: usize, depth: usize) -> Vec<f32> {
        (0..count * depth).map(|index| (index * 7 % 11) as f32 - 5.0).collect()
    }

    fn small_bit_vectors(count: usize, depth: usize) -> Vec<u1x8> {
        (0..count * depth / 8)
            .map(|index| u1x8::new((index * 37 % 251) as u8))
            .collect()
    }

    fn assert_upper_triangle(
        row: &str,
        count: usize,
        output: impl Fn(usize) -> f64,
        expected: impl Fn(usize, usize) -> f64,
    ) {
        for i in 0..count {
            for j in i..count {
                let (actual, expected) = (output(i * count + j), expected(i, j));
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "{row}[{i}, {j}]: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn angulars_symmetric_rows_match_serial_oracle() {
        Capabilities::cpu_enabled().configure_thread().unwrap();
        let (count, depth) = (5, 8);
        let vectors = small_vectors(count, depth);
        let dot = |i: usize, j: usize| -> f64 {
            (0..depth)
                .map(|d| vectors[i * depth + d] * vectors[j * depth + d])
                .sum::<f32>() as f64
        };
        let expected = |i: usize, j: usize| 1.0 - dot(i, j) / (dot(i, i) * dot(j, j)).sqrt();

        let tensor = Tensor::<f32>::from_slice(&vectors, &[count, depth]).unwrap();
        let mut numkong = Tensor::<f64>::zeros(&[count, count]).unwrap();
        tensor.angulars_symmetric_into(&mut numkong).unwrap();
        assert_upper_triangle("numkong", count, |index| numkong.as_slice()[index], &expected);

        let matrix = Array2::<f32>::from_shape_vec((count, depth), vectors.clone()).unwrap();
        let mut ndarray_output = vec![0.0f32; count * count];
        angulars_from_gram(&matrix.dot(&matrix.t()), count, &mut ndarray_output);
        assert_upper_triangle("ndarray", count, |index| ndarray_output[index] as f64, &expected);

        let matrix = DMatrix::<f32>::from_row_slice(count, depth, &vectors);
        let mut nalgebra_output = vec![0.0f32; count * count];
        angulars_from_gram(&(&matrix * &matrix.transpose()), count, &mut nalgebra_output);
        assert_upper_triangle("nalgebra", count, |index| nalgebra_output[index] as f64, &expected);
    }

    #[test]
    fn euclideans_symmetric_rows_match_serial_oracle() {
        Capabilities::cpu_enabled().configure_thread().unwrap();
        let (count, depth) = (5, 8);
        let vectors = small_vectors(count, depth);
        let expected = |i: usize, j: usize| -> f64 {
            let squared: f32 = (0..depth)
                .map(|d| (vectors[i * depth + d] - vectors[j * depth + d]).powi(2))
                .sum();
            (squared as f64).sqrt()
        };

        let tensor = Tensor::<f32>::from_slice(&vectors, &[count, depth]).unwrap();
        let mut numkong = Tensor::<f64>::zeros(&[count, count]).unwrap();
        tensor.euclideans_symmetric_into(&mut numkong).unwrap();
        assert_upper_triangle("numkong", count, |index| numkong.as_slice()[index], &expected);

        let matrix = Array2::<f32>::from_shape_vec((count, depth), vectors.clone()).unwrap();
        let mut ndarray_output = vec![0.0f32; count * count];
        euclideans_from_gram(&matrix.dot(&matrix.t()), count, &mut ndarray_output);
        assert_upper_triangle("ndarray", count, |index| ndarray_output[index] as f64, &expected);

        let matrix = DMatrix::<f32>::from_row_slice(count, depth, &vectors);
        let mut nalgebra_output = vec![0.0f32; count * count];
        euclideans_from_gram(&(&matrix * &matrix.transpose()), count, &mut nalgebra_output);
        assert_upper_triangle("nalgebra", count, |index| nalgebra_output[index] as f64, &expected);
    }

    #[test]
    fn hammings_symmetric_rows_match_serial_oracle() {
        Capabilities::cpu_enabled().configure_thread().unwrap();
        // 72 bits leave the second 64-bit word of every row partly zero-padded
        let (count, depth) = (5, 72);
        let vectors = small_bit_vectors(count, depth);
        let bit = |row: usize, dimension: usize| vectors[(row * depth + dimension) / 8].bits() >> (dimension % 8) & 1;
        let expected = |i: usize, j: usize| (0..depth).filter(|&d| bit(i, d) != bit(j, d)).count() as f64;

        let tensor = Tensor::<u1x8>::from_slice(&vectors, &[count, depth]).unwrap();
        let mut numkong = Tensor::<u32>::zeros(&[count, count]).unwrap();
        tensor.hammings_symmetric_into(&mut numkong).unwrap();
        assert_upper_triangle("numkong", count, |index| numkong.as_slice()[index] as f64, &expected);

        let mut baseline = vec![0u32; count * count];
        baseline_hammings_symmetric(&bit_words(&vectors, count), count, &mut baseline);
        assert_upper_triangle("baseline", count, |index| baseline[index] as f64, &expected);
    }

    #[test]
    fn jaccards_symmetric_rows_match_serial_oracle() {
        Capabilities::cpu_enabled().configure_thread().unwrap();
        let (count, depth) = (5, 72);
        let vectors = small_bit_vectors(count, depth);
        let bit = |row: usize, dimension: usize| vectors[(row * depth + dimension) / 8].bits() >> (dimension % 8) & 1;
        let expected = |i: usize, j: usize| {
            let intersection = (0..depth).filter(|&d| bit(i, d) & bit(j, d) == 1).count() as f64;
            let union = (0..depth).filter(|&d| bit(i, d) | bit(j, d) == 1).count() as f64;
            1.0 - intersection / union
        };

        let tensor = Tensor::<u1x8>::from_slice(&vectors, &[count, depth]).unwrap();
        let mut numkong = Tensor::<f32>::zeros(&[count, count]).unwrap();
        tensor.jaccards_symmetric_into(&mut numkong).unwrap();
        assert_upper_triangle("numkong", count, |index| numkong.as_slice()[index] as f64, &expected);

        let mut baseline = vec![0.0f32; count * count];
        baseline_jaccards_symmetric(&bit_words(&vectors, count), count, &mut baseline);
        assert_upper_triangle("baseline", count, |index| baseline[index] as f64, &expected);
    }
}

// endregion

// region: Main

fn main() {
    let settings = Settings::read("similarities");
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    propagate_thread_count(settings.threads);
    #[cfg(feature = "cuda")]
    let devices = gpu_devices(&settings);
    let mut criterion = numwars::configure_criterion(&settings);
    bench_angulars(&mut criterion, &settings);
    bench_euclideans(&mut criterion, &settings);
    bench_hammings(&mut criterion, &settings);
    bench_jaccards(&mut criterion, &settings);
    #[cfg(feature = "cuda")]
    bench_similarities_gpu(&mut criterion, &settings, &devices);
    criterion.final_summary();
}

// endregion
