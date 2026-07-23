// Feature unification can switch on `ndarray/blas` for every binary in a `--features all` build,
// so the OpenBLAS provider must be linked wherever it is enabled — declaring it here covers all
// benchmarks at once.
#[cfg(feature = "blas-src")]
extern crate blas_src;
#[cfg(feature = "openblas-src")]
extern crate openblas_src;

use std::env;
use std::str::FromStr;

use criterion::Criterion;
use numkong::{bf16, e2m3, e3m2, e4m3, e5m2, f16};

#[cfg(feature = "rand")]
use rand::Rng;

#[cfg(feature = "forkunion")]
use forkunion::{ThreadPool, Topology};
#[cfg(feature = "forkunion")]
use numkong::{Dots, DotsPackedMatrix, TensorRef};

#[cfg(feature = "openblas-src")]
use std::ffi::c_int;

// region: Environment Variable Helpers
//
// Standardized functions for fetching environment variables consistently.
// Use these instead of raw env::var() calls throughout the codebase.

/// Get an optional environment variable, returning None if not set.
pub fn get_env(name: &str) -> Option<String> {
    env::var(name).ok()
}

/// Get an environment variable parsed to a type, with a default value.
/// Returns the default if the variable is not set or cannot be parsed.
pub fn get_env_parsed<T: FromStr>(name: &str, default: T) -> T {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// endregion

// region: Baseline Conversion

/// Trait for converting element types to accumulator types in baselines.
/// Replaces `NumCast` for types (like mini-floats) that can't implement it.
#[allow(dead_code)]
pub trait BaselineConvert<Acc>: Copy {
    fn to_acc(self) -> Acc;
}

impl BaselineConvert<f32> for f32 {
    fn to_acc(self) -> f32 {
        self
    }
}
impl BaselineConvert<f64> for f64 {
    fn to_acc(self) -> f64 {
        self
    }
}
impl BaselineConvert<i32> for i8 {
    fn to_acc(self) -> i32 {
        self as i32
    }
}
impl BaselineConvert<i32> for u8 {
    fn to_acc(self) -> i32 {
        self as i32
    }
}
impl BaselineConvert<u32> for u8 {
    fn to_acc(self) -> u32 {
        self as u32
    }
}
impl BaselineConvert<f32> for f16 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for bf16 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for e4m3 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for e5m2 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for e2m3 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for e3m2 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}

// endregion

// region: Data Generation

/// Generic random number generation for numeric types.
/// Fills a mutable slice with random values in-place (zero allocation).
///
/// Values are uniformly distributed in the range [-1.0, 1.0] for floats,
/// or full range for integer types.
#[cfg(feature = "rand")]
#[allow(dead_code)]
pub fn fill_random<T, R: Rng>(rng: &mut R, data: &mut [T]) {
    let byte_len = data.len() * std::mem::size_of::<T>();
    if byte_len > 0 {
        // SAFETY: Any bit pattern is acceptable for throughput benchmarks.
        // The byte slice covers exactly the memory of `data`.
        unsafe {
            rng.fill_bytes(std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, byte_len));
        }
    }
}

/// Allocate a `Vec<T>` of `count` elements filled with random bytes.
#[cfg(feature = "rand")]
#[allow(dead_code)]
pub fn generate_random<T>(rng: &mut impl Rng, count: usize) -> Vec<T> {
    let mut data = Vec::<T>::with_capacity(count);
    // SAFETY: fill_random writes every byte before any read can occur.
    unsafe { data.set_len(count) };
    fill_random(rng, &mut data);
    data
}

// endregion

// region: Dimension Helpers

/// Get matrix output width in C = A @ B.T where C is height×width.
/// Used in dots/ module for matrix multiplication benchmarks.
#[allow(dead_code)]
pub fn get_matrix_dims_width() -> usize {
    get_env_parsed("NUMWARS_DIMS_WIDTH", 2048)
}

/// Get matrix output height in C = A @ B.T where C is height×width.
/// Used in dots/ module for matrix multiplication benchmarks.
#[allow(dead_code)]
pub fn get_matrix_dims_height() -> usize {
    get_env_parsed("NUMWARS_DIMS_HEIGHT", 2048)
}

/// Get matrix shared dimension in A @ B.T where A is height×depth and B is width×depth.
/// Used in dots/ module for matrix multiplication benchmarks.
#[allow(dead_code)]
pub fn get_matrix_dims_depth() -> usize {
    get_env_parsed("NUMWARS_DIMS_DEPTH", 2048)
}

/// Get tensor dimensions for elementwise operations.
/// Used in each/ module for elementwise operation benchmarks.
#[allow(dead_code)]
pub fn get_tensor_dims() -> usize {
    get_env_parsed("NUMWARS_DIMS", 1_000_000)
}

/// Get vector dimensions for similarity operations.
/// Also doubles as the row count for batched distance-matrix benchmarks.
#[allow(dead_code)]
pub fn get_vector_dims() -> usize {
    get_env_parsed("NUMWARS_DIMS", 2048)
}

/// Get thread count for parallel benchmarks.
/// Defaults to 1 (single-threaded). Set to 0 to use all logical CPUs.
#[allow(dead_code)]
pub fn get_thread_count() -> usize {
    let n = get_env_parsed("NUMWARS_THREADS", 1);
    if n == 0 { num_cpus::get() } else { n }
}

// endregion

// region: Parallelism Helpers
//
// Shared by every benchmark that drives a NumKong `*_parallel` entry point, so pool setup and
// packing look identical across binaries and only `NUMWARS_THREADS` decides the shape of a run.

/// Process-wide CPU topology, probed once; ForkUnion pools spawn onto this shared handle.
#[cfg(feature = "forkunion")]
#[allow(dead_code)]
pub fn topology() -> &'static Topology {
    static TOPOLOGY: std::sync::OnceLock<Topology> = std::sync::OnceLock::new();
    TOPOLOGY.get_or_init(|| Topology::new().expect("Failed to probe CPU topology"))
}

/// Spawn a pool sized by `NUMWARS_THREADS`, or `None` when the run is single-threaded.
/// No warm-up pass is needed: NumKong's parallel kernels call `configure_thread` on every worker.
#[cfg(feature = "forkunion")]
#[allow(dead_code)]
pub fn try_spawn_pool() -> Option<ThreadPool> {
    let threads = get_thread_count();
    if threads <= 1 {
        return None;
    }
    let pool = ThreadPool::try_spawn(topology(), threads).expect("Failed to spawn thread pool");
    Some(pool)
}

/// Pack the B operand into a [`DotsPackedMatrix`], on the pool when one is running.
#[cfg(feature = "forkunion")]
#[allow(dead_code)]
pub fn pack_dots_matrix<Scalar, Matrix, const MAX_RANK: usize>(
    b: &Matrix,
    pool: Option<&mut ThreadPool>,
) -> DotsPackedMatrix<Scalar>
where
    Scalar: Dots + Clone + Send + Sync,
    Matrix: TensorRef<Scalar, MAX_RANK>,
{
    match pool {
        Some(pool) => DotsPackedMatrix::try_pack_parallel(b, pool).expect("Failed to pack B"),
        None => DotsPackedMatrix::try_pack(b).expect("Failed to pack B"),
    }
}

#[cfg(feature = "openblas-src")]
extern "C" {
    fn openblas_set_num_threads(num_threads: c_int);
}

/// Propagate NUMWARS_THREADS to competitor backends at runtime.
///
/// OpenBLAS ignores env vars set after library init, so we call the C API directly.
/// matrixmultiply reads MATMUL_NUM_THREADS lazily on first use, so env var works.
#[cfg(feature = "openblas-src")]
#[allow(dead_code)]
pub fn propagate_thread_count() {
    let threads = get_thread_count();
    unsafe { openblas_set_num_threads(threads as c_int) };
    std::env::set_var("MATMUL_NUM_THREADS", threads.to_string());
}

// endregion

// region: Benchmark Filtering

/// Check if a benchmark should run based on NUMWARS_FILTER regex.
///
/// This is the ONLY filtering mechanism. All benchmarks generate all combinations
/// of dtype × operation/metric by default. Use NUMWARS_FILTER to selectively run
/// specific benchmarks via regex matching on the full benchmark name.
///
/// Example benchmark names:
/// - "similarity/angular/f32"
/// - "each/add/f64"
/// - "dots/f32/1024x1024x1024"
///
/// Example filters:
/// - NUMWARS_FILTER="f32" → only f32 benchmarks
/// - NUMWARS_FILTER="angular|dot" → only angular and dot metrics
/// - NUMWARS_FILTER="each/add" → only add operations in each module
pub fn should_run_benchmark(name: &str) -> bool {
    if let Some(filter) = get_env("NUMWARS_FILTER") {
        // Use regex matching
        if let Ok(re) = regex::Regex::new(&filter) {
            return re.is_match(name);
        } else {
            // Fallback to substring match if regex is invalid
            eprintln!("Warning: Invalid regex in NUMWARS_FILTER, using substring match");
            return name.contains(&filter);
        }
    }

    // No filter set = run all benchmarks
    true
}

// endregion

// region: Criterion Configuration

/// Configure Criterion with standardized settings from environment variables.
pub fn configure_criterion() -> Criterion {
    use std::time::Duration;

    let warmup_time = get_env_parsed("NUMWARS_WARMUP_SECONDS", 3.0);
    let profile_time = get_env_parsed("NUMWARS_PROFILE_SECONDS", 10.0);
    let sample_size = get_env_parsed("NUMWARS_SAMPLE_SIZE", 50);

    Criterion::default()
        .warm_up_time(Duration::from_secs_f64(warmup_time))
        .measurement_time(Duration::from_secs_f64(profile_time))
        .sample_size(sample_size)
        .configure_from_args()
}

// endregion
