//! Shared harness for the NumWars Rust suites. Mirrors `numwars.py`.
//!
//! Variable                Default  Meaning
//! NUMWARS_FILTER          none     Regex over benchmark names, or a substring if not a regex
//! NUMWARS_SEED            42       32-bit seed for random inputs, or `random`
//! NUMWARS_WARMUP          1s       Warm-up per benchmark, like `200ms` or `1s`
//! NUMWARS_TIME_LIMIT      10s      Measurement time per benchmark, like `200ms` or `10s`
//! NUMWARS_BATCH_PER_CORE  2048     Items per call, times the threads: elements, rows, points
//! NUMWARS_THREADS         1        Threads for NumKong and competitors, `0` for all cores
//! NUMWARS_DIMS            1536     Vector length
//! NUMWARS_DIMS_HEIGHT     1024     Rows of `C = A @ B.T` in GEMM-shaped suites
//! NUMWARS_DIMS_WIDTH      128      Columns of `C = A @ B.T` in GEMM-shaped suites
//! NUMWARS_DIMS_DEPTH      1536     Shared dimension of `A` and `B` in GEMM-shaped suites
//! NUMWARS_SAMPLE_SIZE     50       Criterion samples per benchmark

// Feature unification can switch on `ndarray/blas` for every binary in a `--features all` build,
// so the OpenBLAS provider must be linked wherever it is enabled.
#[cfg(feature = "blas-src")]
extern crate blas_src;
#[cfg(feature = "openblas-src")]
extern crate openblas_src;

use std::env;
use std::time::Duration;

#[cfg(feature = "openblas-src")]
use std::ffi::c_int;

use criterion::Criterion;
use numkong::{bf16, e2m3, e3m2, e4m3, e5m2, f16};

#[cfg(feature = "forkunion")]
use forkunion::{ThreadPool, Topology};
#[cfg(feature = "forkunion")]
use numkong::{Dots, DotsPackedMatrix, TensorRef};

// region: Environment variables

/// Reads `name`, or `None` when it is unset or empty.
pub fn env_text(name: &str) -> Option<String> {
    env::var(name).ok().filter(|text| !text.is_empty())
}

/// Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 if it does not parse.
pub fn env_parsed<T>(name: &str, fallback: T, parse: impl FnOnce(&str) -> Option<T>, expected: &str) -> T {
    let Some(text) = env_text(name) else {
        return fallback;
    };
    parse(&text).unwrap_or_else(|| {
        eprintln!("{name}=\"{text}\" does not parse, expected {expected}");
        std::process::exit(1)
    })
}

/// Reads a positive count like `128`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_count(name: &str, fallback: usize) -> usize {
    env_parsed(name, fallback, parse_count, "a positive count")
}

/// Reads a duration like `200ms` or `10s`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_duration(name: &str, fallback: Duration) -> Duration {
    env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s")
}

/// Reads a 32-bit seed or `random`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_seed(name: &str, fallback: u32) -> u32 {
    env_parsed(name, fallback, parse_seed, "an unsigned integer or random")
}

/// Parses a 32-bit unsigned integer, or `random` as 32 bits from the OS entropy source.
pub fn parse_seed(text: &str) -> Option<u32> {
    if text == "random" {
        use std::hash::{BuildHasher, Hasher};

        return Some(std::hash::RandomState::new().build_hasher().finish() as u32);
    }
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok()).flatten()
}

/// Parses a thread count like `8`, or `0` as every available core.
pub fn parse_threads(text: &str) -> Option<usize> {
    match text {
        "0" => Some(std::thread::available_parallelism().map_or(1, usize::from)),
        _ => parse_count(text),
    }
}

/// Parses a positive whole number in ASCII digits, like `128`; zero is `None`.
pub fn parse_count(text: &str) -> Option<usize> {
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok()).flatten().filter(|&count| count != 0)
}

/// Parses a duration like `200ms` or `10s`; a bare number, a fraction or zero is `None`.
pub fn parse_duration(text: &str) -> Option<Duration> {
    match text.strip_suffix("ms") {
        Some(count) => parse_count(count).map(|count| Duration::from_millis(count as u64)),
        None => parse_count(text.strip_suffix('s')?).map(|count| Duration::from_secs(count as u64)),
    }
}

/// Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`.
pub fn spell_duration(duration: Duration) -> String {
    let milliseconds = duration.as_millis();
    match milliseconds % 1000 {
        0 => format!("{}s", milliseconds / 1000),
        _ => format!("{milliseconds}ms"),
    }
}

// endregion

// region: Settings

/// Every `NUMWARS_*` setting, read once at the top of `main` and passed down. Mirrors the Python
/// `Settings` field for field, plus `sample_size`.
pub struct Settings {
    pub filter: String,
    pub filter_pattern: Option<regex::Regex>,
    pub seed: u32,
    pub warmup: Duration,
    pub time_limit: Duration,
    pub batch_per_core: usize,
    pub threads: usize,
    pub dims: usize,
    pub dims_height: usize,
    pub dims_width: usize,
    pub dims_depth: usize,
    pub sample_size: usize,
}

impl Settings {
    /// Reads every `NUMWARS_*` variable, exiting with status 1 on the first that does not parse,
    /// and prints each as "- Name: value".
    pub fn read() -> Self {
        let filter = env_text("NUMWARS_FILTER").unwrap_or_default();
        let settings = Settings {
            filter_pattern: (!filter.is_empty()).then(|| regex::Regex::new(&filter).ok()).flatten(),
            filter,
            seed: env_seed("NUMWARS_SEED", 42),
            warmup: env_duration("NUMWARS_WARMUP", Duration::from_secs(1)),
            time_limit: env_duration("NUMWARS_TIME_LIMIT", Duration::from_secs(10)),
            batch_per_core: env_count("NUMWARS_BATCH_PER_CORE", 2048),
            threads: env_parsed("NUMWARS_THREADS", 1, parse_threads, "a count, 0 for all cores"),
            dims: env_count("NUMWARS_DIMS", 1536),
            dims_height: env_count("NUMWARS_DIMS_HEIGHT", 1024),
            dims_width: env_count("NUMWARS_DIMS_WIDTH", 128),
            dims_depth: env_count("NUMWARS_DIMS_DEPTH", 1536),
            sample_size: env_count("NUMWARS_SAMPLE_SIZE", 50),
        };
        settings.print();
        settings
    }

    /// Prints every setting as "- Name: value", in the grammar it is read in.
    pub fn print(&self) {
        println!("- Seed: {}", self.seed);
        println!(
            "- Filter: {}",
            if self.filter.is_empty() { "none" } else { &self.filter }
        );
        println!("- Warm-up: {}", spell_duration(self.warmup));
        println!("- Time limit: {}", spell_duration(self.time_limit));
        println!("- Batch per core: {}", self.batch_per_core);
        println!("- Threads: {}", self.threads);
        println!("- Dims: {}", self.dims);
        println!("- Dims height: {}", self.dims_height);
        println!("- Dims width: {}", self.dims_width);
        println!("- Dims depth: {}", self.dims_depth);
        println!("- Sample size: {}", self.sample_size);
    }

    /// Whether `NUMWARS_FILTER` selects the row `name`, as a regex or else as a substring.
    pub fn selects(&self, name: &str) -> bool {
        match &self.filter_pattern {
            Some(pattern) => pattern.is_match(name),
            None => name.contains(&self.filter),
        }
    }

    /// Items one call processes: the batch per core times the threads.
    pub fn batch_size(&self) -> usize {
        self.batch_per_core * self.threads
    }
}

/// Criterion with the warm-up, time limit and sample size from `settings`.
pub fn configure_criterion(settings: &Settings) -> Criterion {
    Criterion::default()
        .warm_up_time(settings.warmup)
        .measurement_time(settings.time_limit)
        .sample_size(settings.sample_size)
        .configure_from_args()
}

// endregion

// region: Baseline Conversion

/// Trait for converting element types to accumulator types in baselines.
/// Replaces `NumCast` for types (like mini-floats) that can't implement it.
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

// region: Parallelism Helpers
//
// Shared by every benchmark that drives a NumKong `*_parallel` entry point, so pool setup and
// packing look identical across binaries and only `NUMWARS_THREADS` decides the shape of a run.

/// Process-wide CPU topology, probed once; ForkUnion pools spawn onto this shared handle.
#[cfg(feature = "forkunion")]
pub fn topology() -> &'static Topology {
    static TOPOLOGY: std::sync::OnceLock<Topology> = std::sync::OnceLock::new();
    TOPOLOGY.get_or_init(|| Topology::new().expect("Failed to probe CPU topology"))
}

/// Spawn a pool of `threads`, or `None` when the run is single-threaded.
/// No warm-up pass is needed: NumKong's parallel kernels call `configure_thread` on every worker.
#[cfg(feature = "forkunion")]
pub fn try_spawn_pool(threads: usize) -> Option<ThreadPool> {
    if threads <= 1 {
        return None;
    }
    let pool = ThreadPool::try_spawn(topology(), threads).expect("Failed to spawn thread pool");
    Some(pool)
}

/// Pack the B operand into a [`DotsPackedMatrix`], on the pool when one is running.
#[cfg(feature = "forkunion")]
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

/// Propagate `threads` to competitor backends at runtime.
///
/// OpenBLAS ignores env vars set after library init, so we call the C API directly.
/// matrixmultiply reads MATMUL_NUM_THREADS lazily on first use, so env var works.
#[cfg(feature = "openblas-src")]
pub fn propagate_thread_count(threads: usize) {
    unsafe { openblas_set_num_threads(threads as c_int) };
    std::env::set_var("MATMUL_NUM_THREADS", threads.to_string());
}

// endregion
