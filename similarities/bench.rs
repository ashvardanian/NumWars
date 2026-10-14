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
    let settings = Settings::read();
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    propagate_thread_count(settings.threads);
    let mut criterion = numwars::configure_criterion(&settings);
    bench_angulars(&mut criterion, &settings);
    bench_euclideans(&mut criterion, &settings);
    bench_hammings(&mut criterion, &settings);
    bench_jaccards(&mut criterion, &settings);
    criterion.final_summary();
}

// endregion
