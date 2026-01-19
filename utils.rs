use std::env;
use std::fmt;
use std::panic;
use std::str::FromStr;

#[cfg(feature = "bench_similarity")]
use criterion::Criterion;
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
use rand::Rng;

// NumKong types for alternative float representations
#[cfg(any(feature = "bench_similarity", feature = "bench_each", feature = "bench_dots"))]
use numkong::{bf16, e4m3, e5m2, f16};

// region: Environment Variable Helpers
//
// Standardized functions for fetching environment variables consistently.
// Use these instead of raw env::var() calls throughout the codebase.

/// Get an optional environment variable, returning None if not set.
#[allow(dead_code)]
pub fn get_env(name: &str) -> Option<String> {
    env::var(name).ok()
}

/// Get an environment variable with a default value.
#[allow(dead_code)]
pub fn get_env_or_default(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

/// Get an environment variable parsed to a type, with a default value.
/// Returns the default if the variable is not set or cannot be parsed.
#[allow(dead_code)]
pub fn get_env_parsed<T: FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Get an optional environment variable parsed to a type.
/// Returns None if the variable is not set or cannot be parsed.
#[allow(dead_code)]
pub fn get_env_parsed_opt<T: FromStr>(name: &str) -> Option<T> {
    env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Get a boolean environment variable.
/// Accepts "1", "true", or "yes" (case-insensitive) as true values.
/// Returns false if not set or set to any other value.
#[allow(dead_code)]
pub fn get_env_bool(name: &str) -> bool {
    env::var(name)
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Parse a comma-separated list of values from an environment variable.
#[allow(dead_code)]
pub fn get_env_list<T: FromStr>(name: &str) -> Vec<T> {
    env::var(name)
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_default()
}

// endregion

// region: Error Handling & Panic Hooks

/// Installs a custom panic hook that formats errors cleanly for CLI usage.
/// Call this at the start of main() before any potential panics.
#[allow(dead_code)]
pub fn install_panic_hook() {
    panic::set_hook(Box::new(|info| {
        let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown error".to_string()
        };

        // Print clean error message
        eprintln!("\nError: {}", message);

        // Only show location in debug builds or if RUST_BACKTRACE is set
        if cfg!(debug_assertions) || get_env("RUST_BACKTRACE").is_some() {
            if let Some(location) = info.location() {
                eprintln!("  at {}:{}", location.file(), location.line());
            }
        }
    }));
}

/// Extension trait for Result that provides clean panic-on-error semantics.
#[allow(dead_code)]
pub trait ResultExt<T> {
    /// Unwrap the result or panic with the Display-formatted error.
    fn unwrap_nice(self) -> T;

    /// Unwrap the result or panic with a custom message and the error.
    fn expect_nice(self, msg: &str) -> T;
}

impl<T, E: fmt::Display> ResultExt<T> for Result<T, E> {
    #[track_caller]
    fn unwrap_nice(self) -> T {
        match self {
            Ok(v) => v,
            Err(e) => panic!("{}", e),
        }
    }

    #[track_caller]
    fn expect_nice(self, msg: &str) -> T {
        match self {
            Ok(v) => v,
            Err(e) => panic!("{}: {}", msg, e),
        }
    }
}

/// Extension trait for Option that provides clean panic-on-none semantics.
#[allow(dead_code)]
pub trait OptionExt<T> {
    /// Unwrap the option or panic with a custom message.
    fn expect_nice(self, msg: &str) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    #[track_caller]
    fn expect_nice(self, msg: &str) -> T {
        match self {
            Some(v) => v,
            None => panic!("{}", msg),
        }
    }
}

/// Errors that can occur during benchmark operations.
#[derive(Debug)]
pub enum BenchmarkError {
    /// Invalid configuration parameter.
    InvalidConfig { param: String, reason: String },
    /// Data generation failed.
    DataGeneration { reason: String },
    /// Unsupported data type for the operation.
    UnsupportedType { dtype: String, operation: String },
    /// Matrix dimension mismatch.
    DimensionMismatch { expected: String, got: String },
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchmarkError::InvalidConfig { param, reason } => {
                write!(f, "Invalid configuration for '{}': {}", param, reason)
            }
            BenchmarkError::DataGeneration { reason } => {
                write!(f, "Failed to generate data: {}", reason)
            }
            BenchmarkError::UnsupportedType { dtype, operation } => {
                write!(
                    f,
                    "Data type '{}' is not supported for operation '{}'",
                    dtype, operation
                )
            }
            BenchmarkError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for BenchmarkError {}

// endregion

// region: Data Generation

/// Generic random number generation for numeric types.
/// Fills a mutable slice with random values in-place (zero allocation).
///
/// Values are uniformly distributed in the range [-1.0, 1.0] for floats,
/// or full range for integer types.
///
/// # Examples
/// ```
/// let mut data = vec![0.0f32; 1000];
/// fill_random(&mut rng, &mut data);
/// ```
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
#[allow(dead_code)]
pub fn fill_random<T, R: Rng>(rng: &mut R, data: &mut [T])
where
    T: RandomFillable,
{
    T::fill_random(rng, data);
}

/// Trait for types that can be randomly generated.
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
pub trait RandomFillable: Sized {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]);
}

// Implementation for f32 - base case
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for f32 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = rng.gen_range(-1.0..1.0);
        }
    }
}

// Implementation for f64 - base case
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for f64 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = rng.gen_range(-1.0..1.0);
        }
    }
}

// Implementation for i8 - full range
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for i8 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = rng.gen();
        }
    }
}

// Implementation for u8 - full range
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for u8 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = rng.gen();
        }
    }
}

// Implementation for NumKong f16 - via f32
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for f16 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = f16::from_f32(rng.gen_range(-1.0..1.0));
        }
    }
}

// Implementation for NumKong bf16 - via f32
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for bf16 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = bf16::from_f32(rng.gen_range(-1.0..1.0));
        }
    }
}

// Implementation for NumKong e4m3 - via f32
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for e4m3 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = e4m3::from_f32(rng.gen_range(-1.0..1.0));
        }
    }
}

// Implementation for NumKong e5m2 - via f32
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl RandomFillable for e5m2 {
    fn fill_random<R: Rng>(rng: &mut R, data: &mut [Self]) {
        for val in data.iter_mut() {
            *val = e5m2::from_f32(rng.gen_range(-1.0..1.0));
        }
    }
}

/// Generate random probability distribution (values sum to 1.0).
/// Fills a mutable slice with normalized values in-place.
///
/// Uses normalized uniform distribution in range [0.01, 1.0].
///
/// # Examples
/// ```
/// let mut dist = vec![0.0f32; 100];
/// fill_random_distribution(&mut rng, &mut dist);
/// assert!((dist.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// ```
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
#[allow(dead_code)]
pub fn fill_random_distribution<T, R: Rng>(rng: &mut R, data: &mut [T])
where
    T: DistributionFillable,
{
    T::fill_distribution(rng, data);
}

/// Trait for types that support probability distribution generation.
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
pub trait DistributionFillable: Sized {
    fn fill_distribution<R: Rng>(rng: &mut R, data: &mut [Self]);
}

// Implementation for f32
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl DistributionFillable for f32 {
    fn fill_distribution<R: Rng>(rng: &mut R, data: &mut [Self]) {
        // Fill with random values in [0.01, 1.0]
        for val in data.iter_mut() {
            *val = rng.gen_range(0.01..1.0);
        }
        // Normalize to sum = 1.0
        let sum: f32 = data.iter().sum();
        for val in data.iter_mut() {
            *val /= sum;
        }
    }
}

// Implementation for f64
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
impl DistributionFillable for f64 {
    fn fill_distribution<R: Rng>(rng: &mut R, data: &mut [Self]) {
        // Fill with random values in [0.01, 1.0]
        for val in data.iter_mut() {
            *val = rng.gen_range(0.01..1.0);
        }
        // Normalize to sum = 1.0
        let sum: f64 = data.iter().sum();
        for val in data.iter_mut() {
            *val /= sum;
        }
    }
}

// Backward compatibility wrappers for existing code
// These will be removed after updating all benchmarks to use fill_random

/// Generate random f32 values in the range [-1.0, 1.0].
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
#[allow(dead_code)]
pub fn generate_random_f32<R: Rng>(rng: &mut R, count: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; count];
    fill_random(rng, &mut data);
    data
}

/// Generate random f64 values in the range [-1.0, 1.0].
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
#[allow(dead_code)]
pub fn generate_random_f64<R: Rng>(rng: &mut R, count: usize) -> Vec<f64> {
    let mut data = vec![0.0f64; count];
    fill_random(rng, &mut data);
    data
}

/// Generate random i8 values in the full range.
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
#[allow(dead_code)]
pub fn generate_random_i8<R: Rng>(rng: &mut R, count: usize) -> Vec<i8> {
    let mut data = vec![0i8; count];
    fill_random(rng, &mut data);
    data
}

/// Generate random u8 values (useful for binary similarity).
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
#[allow(dead_code)]
pub fn generate_random_u8<R: Rng>(rng: &mut R, count: usize) -> Vec<u8> {
    let mut data = vec![0u8; count];
    fill_random(rng, &mut data);
    data
}

// endregion

// region: Benchmark Utilities

/// Calculate throughput in GB/s given bytes processed and duration.
#[allow(dead_code)]
pub fn calculate_throughput(bytes: usize, duration_secs: f64) -> f64 {
    if duration_secs > 0.0 {
        (bytes as f64) / duration_secs / 1e9
    } else {
        0.0
    }
}

/// Calculate operations per second.
#[allow(dead_code)]
pub fn calculate_ops_per_sec(operations: usize, duration_secs: f64) -> f64 {
    if duration_secs > 0.0 {
        (operations as f64) / duration_secs
    } else {
        0.0
    }
}

/// Format operations per second with auto-scaling units.
///
/// Similar to StringWars CUPS (Characters Used Per Second) formatter.
/// Automatically selects appropriate scale: Ops, KiloOps, MegaOps, GigaOps, TeraOps.
///
/// # Examples
/// ```
/// let ops_per_sec = 1_234_567_890.0;
/// assert_eq!(format_ops_per_sec(ops_per_sec), "1.23 GigaOps/s");
/// ```
#[allow(dead_code)]
pub fn format_ops_per_sec(ops_per_sec: f64) -> String {
    if ops_per_sec >= 1e12 {
        format!("{:.2} TeraOps/s", ops_per_sec / 1e12)
    } else if ops_per_sec >= 1e9 {
        format!("{:.2} GigaOps/s", ops_per_sec / 1e9)
    } else if ops_per_sec >= 1e6 {
        format!("{:.2} MegaOps/s", ops_per_sec / 1e6)
    } else if ops_per_sec >= 1e3 {
        format!("{:.2} KiloOps/s", ops_per_sec / 1e3)
    } else {
        format!("{:.2} Ops/s", ops_per_sec)
    }
}

/// Calculate absolute and relative error between expected and actual values.
#[allow(dead_code)]
pub fn calculate_error_f32(expected: f32, actual: f32) -> (f32, f32) {
    let abs_err = (expected - actual).abs();
    let rel_err = if expected.abs() > 1e-9 {
        abs_err / expected.abs()
    } else {
        abs_err
    };
    (abs_err, rel_err)
}

/// Calculate absolute and relative error between expected and actual values (f64).
#[allow(dead_code)]
pub fn calculate_error_f64(expected: f64, actual: f64) -> (f64, f64) {
    let abs_err = (expected - actual).abs();
    let rel_err = if expected.abs() > 1e-15 {
        abs_err / expected.abs()
    } else {
        abs_err
    };
    (abs_err, rel_err)
}

/// Calculate mean absolute error across a vector of results.
#[allow(dead_code)]
pub fn mean_absolute_error_f32(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    let sum: f32 = expected
        .iter()
        .zip(actual.iter())
        .map(|(e, a)| (e - a).abs())
        .sum();
    sum / expected.len() as f32
}

/// Calculate mean absolute error across a vector of results (f64).
#[allow(dead_code)]
pub fn mean_absolute_error_f64(expected: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(expected.len(), actual.len());
    let sum: f64 = expected
        .iter()
        .zip(actual.iter())
        .map(|(e, a)| (e - a).abs())
        .sum();
    sum / expected.len() as f64
}

/// Format a large number with thousands separators.
#[allow(dead_code)]
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

/// Format bytes as a human-readable string (KB, MB, GB, TB).
#[allow(dead_code)]
pub fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let bytes_f = bytes as f64;
    if bytes_f >= TB {
        format!("{:.2} TB", bytes_f / TB)
    } else if bytes_f >= GB {
        format!("{:.2} GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.2} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.2} KB", bytes_f / KB)
    } else {
        format!("{} B", bytes)
    }
}

// endregion

// region: Dimension Helpers

/// Get matrix output width (n in C = A @ B.T where C is m×n).
/// Used in dots/ module for matrix multiplication benchmarks.
#[allow(dead_code)]
pub fn get_matrix_dims_width() -> usize {
    get_env_parsed("NUMWARS_DIMS_WIDTH", 1024)
}

/// Get matrix output height (m in C = A @ B.T where C is m×n).
/// Used in dots/ module for matrix multiplication benchmarks.
#[allow(dead_code)]
pub fn get_matrix_dims_height() -> usize {
    get_env_parsed("NUMWARS_DIMS_HEIGHT", 1024)
}

/// Get matrix shared dimension (k in A @ B.T where A is m×k and B is n×k).
/// Used in dots/ module for matrix multiplication benchmarks.
#[allow(dead_code)]
pub fn get_matrix_dims_depth() -> usize {
    get_env_parsed("NUMWARS_DIMS_DEPTH", 1024)
}

/// Get tensor dimensions for elementwise operations.
/// Used in each/ module for elementwise operation benchmarks.
#[allow(dead_code)]
pub fn get_tensor_dims() -> usize {
    get_env_parsed("NUMWARS_DIMS", 1_000_000)
}

/// Get vector dimensions for similarity operations.
/// Used in similarity/ module for pairwise vector similarity benchmarks.
#[allow(dead_code)]
pub fn get_vector_dims() -> usize {
    get_env_parsed("NUMWARS_DIMS", 1536)
}

/// Get batch size for similarity operations.
/// Used in similarity/ module to determine number of vector pairs to process.
#[allow(dead_code)]
pub fn get_batch_size() -> usize {
    get_env_parsed("NUMWARS_BATCH_SIZE", 1000)
}

/// Get thread count for parallel benchmarks.
/// Defaults to the number of logical CPUs on the system.
#[allow(dead_code)]
pub fn get_thread_count() -> usize {
    get_env_parsed("NUMWARS_THREADS", num_cpus::get())
}

/// Check if a benchmark should run based on NUMWARS_FILTER regex.
///
/// This is the ONLY filtering mechanism. All benchmarks generate all combinations
/// of dtype × operation/metric by default. Use NUMWARS_FILTER to selectively run
/// specific benchmarks via regex matching on the full benchmark name.
///
/// Example benchmark names:
/// - "similarity/cosine/f32"
/// - "each/add/f64"
/// - "dots/numkong/f32/1024x1024x1024/8t"
///
/// Example filters:
/// - NUMWARS_FILTER="f32" → only f32 benchmarks
/// - NUMWARS_FILTER="cosine|dot" → only cosine and dot metrics
/// - NUMWARS_FILTER="each/add" → only add operations in each module
#[allow(dead_code)]
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
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots"
))]
#[allow(dead_code)]
pub fn configure_criterion() -> Criterion {
    use criterion::*;
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


// region: Testing Utilities

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_throughput() {
        let throughput = calculate_throughput(1_000_000_000, 1.0);
        assert!((throughput - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

}

// endregion
