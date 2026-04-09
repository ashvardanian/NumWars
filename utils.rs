use std::env;
use std::fmt;
use std::panic;
use std::str::FromStr;

#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots",
    feature = "bench_geospatial",
    feature = "bench_similarities",
    feature = "bench_maxsim",
    feature = "bench_reduce",
    feature = "bench_mesh"
))]
use criterion::Criterion;
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots",
    feature = "bench_geospatial",
    feature = "bench_similarities",
    feature = "bench_maxsim",
    feature = "bench_reduce",
    feature = "bench_mesh"
))]
use rand::Rng;

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
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
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
impl BaselineConvert<f32> for numkong::f16 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for numkong::bf16 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for numkong::e4m3 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for numkong::e5m2 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for numkong::e2m3 {
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl BaselineConvert<f32> for numkong::e3m2 {
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
///
/// # Examples
/// ```
/// let mut data = vec![0.0f32; 1000];
/// fill_random(&mut rng, &mut data);
/// ```
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots",
    feature = "bench_geospatial",
    feature = "bench_similarities",
    feature = "bench_maxsim",
    feature = "bench_reduce",
    feature = "bench_mesh"
))]
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
    feature = "bench_dots",
    feature = "bench_geospatial",
    feature = "bench_similarities",
    feature = "bench_maxsim",
    feature = "bench_reduce",
    feature = "bench_mesh"
))]
#[allow(dead_code)]
pub fn fill_random_distribution<T, R: Rng>(rng: &mut R, data: &mut [T]) {
    fill_random(rng, data);
}

/// Allocate a `Vec<T>` of `count` elements filled with random bytes.
#[cfg(any(
    feature = "bench_similarity",
    feature = "bench_each",
    feature = "bench_dots",
    feature = "bench_geospatial",
    feature = "bench_similarities",
    feature = "bench_maxsim",
    feature = "bench_reduce",
    feature = "bench_mesh"
))]
#[allow(dead_code)]
pub fn generate_random<T>(rng: &mut impl Rng, count: usize) -> Vec<T> {
    let mut data = Vec::<T>::with_capacity(count);
    // SAFETY: fill_random writes every byte before any read can occur.
    unsafe { data.set_len(count) };
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

/// Format scalar operations per second with auto-scaling units.
///
/// Similar to StringWars CUPS (Characters Used Per Second) formatter.
/// Automatically selects appropriate scale: SO/s, KSO/s, MSO/s, GSO/s, TSO/s.
///
/// # Examples
/// ```
/// let ops_per_sec = 1_234_567_890.0;
/// assert_eq!(format_ops_per_sec(ops_per_sec), "1.23 GSO/s");
/// ```
#[allow(dead_code)]
pub fn format_ops_per_sec(ops_per_sec: f64) -> String {
    if ops_per_sec >= 1e12 {
        format!("{:.2} TSO/s", ops_per_sec / 1e12)
    } else if ops_per_sec >= 1e9 {
        format!("{:.2} GSO/s", ops_per_sec / 1e9)
    } else if ops_per_sec >= 1e6 {
        format!("{:.2} MSO/s", ops_per_sec / 1e6)
    } else if ops_per_sec >= 1e3 {
        format!("{:.2} KSO/s", ops_per_sec / 1e3)
    } else {
        format!("{:.2} SO/s", ops_per_sec)
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
    let sum: f32 = expected.iter().zip(actual.iter()).map(|(e, a)| (e - a).abs()).sum();
    sum / expected.len() as f32
}

/// Calculate mean absolute error across a vector of results (f64).
#[allow(dead_code)]
pub fn mean_absolute_error_f64(expected: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(expected.len(), actual.len());
    let sum: f64 = expected.iter().zip(actual.iter()).map(|(e, a)| (e - a).abs()).sum();
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
/// Used in similarity/ module for pairwise vector similarity benchmarks.
#[allow(dead_code)]
pub fn get_vector_dims() -> usize {
    get_env_parsed("NUMWARS_DIMS", 2048)
}

/// Alias for get_vector_dims() — kept for backward compatibility.
#[allow(dead_code)]
pub fn get_batch_size() -> usize {
    get_vector_dims()
}

/// Get thread count for parallel benchmarks.
/// Defaults to 1 (single-threaded). Set to 0 to use all logical CPUs.
#[allow(dead_code)]
pub fn get_thread_count() -> usize {
    let n = get_env_parsed("NUMWARS_THREADS", 1);
    if n == 0 { num_cpus::get() } else { n }
}

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
    feature = "bench_dots",
    feature = "bench_geospatial",
    feature = "bench_similarities",
    feature = "bench_maxsim",
    feature = "bench_reduce",
    feature = "bench_mesh"
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
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn throughput_1gb_in_1s() {
        let throughput = calculate_throughput(1_000_000_000, 1.0);
        assert!((throughput - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bytes_formatting_scales() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}

// endregion
