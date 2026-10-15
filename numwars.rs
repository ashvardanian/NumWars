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
//! NUMWARS_DIMS_HEIGHT     1024     Rows of `C = A × Bᵀ` in GEMM-shaped suites
//! NUMWARS_DIMS_WIDTH      128      Columns of `C = A × Bᵀ` in GEMM-shaped suites
//! NUMWARS_DIMS_DEPTH      1536     Shared dimension of `A` and `B` in GEMM-shaped suites
//! NUMWARS_SHAPE_POINTS    32       Points per cloud in the shape-distance suite
//! NUMWARS_DEVICES         cuda:0   CUDA devices in GPU suites, like `cuda:0,cuda:1`, one row per device
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

/// Suites that put rows on a GPU, and so accept `NUMWARS_DEVICES`.
pub const GPU_SUITES: [&str; 3] = ["dots", "similarities", "attention"];

/// Every `NUMWARS_*` setting, read once at the top of `main` and passed down. Mirrors the Python
/// `Settings` field for field, plus `sample_size`.
pub struct Settings {
    pub devices: Option<Vec<usize>>,
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
    pub shape_points: usize,
    pub sample_size: usize,
}

impl Settings {
    /// Reads every `NUMWARS_*` variable `suite` uses, exiting with status 1 on the first that does
    /// not parse, and prints each as "- Name: value".
    pub fn read(suite: &str) -> Self {
        let devices = env_text("NUMWARS_DEVICES").map(|selection| {
            let mut ordinals = Vec::new();
            for item in selection.split(',') {
                let digits = item.strip_prefix("cuda:").filter(|text| !text.is_empty());
                let digits = digits.filter(|text| text.bytes().all(|byte| byte.is_ascii_digit()));
                let Some(ordinal) = digits.and_then(|text| text.parse().ok()) else {
                    eprintln!("NUMWARS_DEVICES requires CUDA selectors such as cuda:0,cuda:1");
                    std::process::exit(1)
                };
                // A repeated ordinal would name two rows identically, so the first wins.
                if !ordinals.contains(&ordinal) {
                    ordinals.push(ordinal);
                }
            }
            ordinals
        });
        if devices.is_some() && !GPU_SUITES.contains(&suite) {
            eprintln!("NUMWARS_DEVICES is unsupported by the {suite} suite, which has no GPU rows");
            std::process::exit(1);
        }
        if devices.is_some() && !cfg!(feature = "cuda") {
            eprintln!("NUMWARS_DEVICES needs the {suite} suite built with the `cuda` feature");
            std::process::exit(1);
        }
        let filter = env_text("NUMWARS_FILTER").unwrap_or_default();
        let settings = Settings {
            devices,
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
            shape_points: env_count("NUMWARS_SHAPE_POINTS", 32),
            sample_size: env_count("NUMWARS_SAMPLE_SIZE", 50),
        };
        settings.print();
        settings
    }

    /// Prints every setting as "- Name: value", in the grammar it is read in.
    pub fn print(&self) {
        if let Some(devices) = &self.devices {
            let selectors: Vec<String> = devices.iter().map(|ordinal| format!("cuda:{ordinal}")).collect();
            println!("- Devices: {}", selectors.join(","));
        }
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
        println!("- Shape points: {}", self.shape_points);
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
    let pool = ThreadPool::spawn(topology(), threads).expect("Failed to spawn thread pool");
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
        Some(pool) => DotsPackedMatrix::new_parallel(b, pool).expect("Failed to pack B"),
        None => DotsPackedMatrix::new(b).expect("Failed to pack B"),
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

// region: GPU Devices

#[cfg(feature = "cuda")]
pub use gpu::*;

/// The GPU rows' devices, operands and cuBLASLt rival. NumKong's Rust verbs dispatch with
/// `Capabilities::CPUS` and a null stream, so these reach its CUDA kernels through the C dispatch
/// it exports, `nk_find_kernel_punned`, with one device's mask and stream. Every operand lives in
/// unified memory, which the host fills and checks.
#[cfg(feature = "cuda")]
mod gpu {
    use std::ffi::{c_int, c_void};

    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublaslt::{result as cublaslt, sys};
    use numkong::{bf16, e2m1x2, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u4x2, Capabilities, Dots, StorageElement};
    use numkong::{Tensor, UnifiedAllocator};
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    use super::Settings;

    extern "C" {
        fn nk_find_kernel_punned(
            kind: c_int,
            dtype: c_int,
            capabilities: u64,
            kernel: *mut Option<unsafe extern "C" fn()>,
            capability: *mut u64,
        ) -> c_int;
    }

    // C's `nk_dots_pack_size_punned_t`, `nk_dots_pack_punned_t`, `nk_dots_packed_punned_t` and
    // `nk_dots_symmetric_punned_t`, which the angular and Euclidean kernels share.
    type PackSizeKernel = unsafe extern "C" fn(usize, usize, *mut usize) -> c_int;
    type PackKernel =
        unsafe extern "C" fn(*const c_void, usize, usize, usize, *mut c_void, usize, usize, *mut c_void) -> c_int;
    type PackedKernel = unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *mut c_void,
        usize,
        usize,
        usize,
        usize,
        usize,
        *mut c_void,
    ) -> c_int;
    type SymmetricKernel = unsafe extern "C" fn(
        *const c_void,
        usize,
        usize,
        usize,
        *mut c_void,
        usize,
        usize,
        usize,
        *mut c_void,
    ) -> c_int;

    // C's `nk_attention_pack_size_punned_t`, `nk_attention_pack_punned_t` and `nk_attention_packed_punned_t`.
    type AttentionPackSizeKernel = unsafe extern "C" fn(usize, usize, usize, usize, *mut usize) -> c_int;
    type AttentionPackKernel = unsafe extern "C" fn(
        usize,
        usize,
        *const u32,
        *const u32,
        usize,
        *const c_void,
        usize,
        *const c_void,
        usize,
        *mut c_void,
        usize,
        usize,
        *mut c_void,
    ) -> c_int;
    type AttentionPackedKernel = unsafe extern "C" fn(
        usize,
        usize,
        usize,
        *const u32,
        usize,
        f32,
        usize,
        usize,
        *const c_void,
        usize,
        *const c_void,
        *mut c_void,
        usize,
        *mut f32,
        usize,
        usize,
        *mut c_void,
    ) -> c_int;

    /// Workspace offered to cuBLASLt, NVIDIA's recommendation from Hopper on.
    const WORKSPACE_BYTES: usize = 32 << 20;

    /// What a GPU row computes between two rows, which picks its packed and symmetric kernels.
    #[derive(Clone, Copy)]
    pub enum GpuMetric {
        Dots,
        Angulars,
        Euclideans,
    }

    impl GpuMetric {
        /// C's `nk_kernel_kind_t` of the packed and of the symmetric kernel.
        fn kinds(self) -> (u8, u8) {
            match self {
                GpuMetric::Dots => (b'G', b'y'),
                GpuMetric::Angulars => (b'N', b'n'),
                GpuMetric::Euclideans => (b'E', b'D'),
            }
        }
    }

    /// The cuBLASLt types of one GEMM: its operands', its output's, its accumulator's and its scalars'.
    #[derive(Clone, Copy)]
    pub struct CublasLtTypes {
        pub input: sys::cudaDataType,
        pub output: sys::cudaDataType,
        pub compute: sys::cublasComputeType_t,
        pub scale: sys::cudaDataType,
    }

    /// Floats multiplied into F32 outputs, accumulating in F32.
    const fn into_f32(input: sys::cudaDataType) -> Option<CublasLtTypes> {
        Some(CublasLtTypes {
            input,
            output: sys::cudaDataType::CUDA_R_32F,
            compute: sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            scale: sys::cudaDataType::CUDA_R_32F,
        })
    }

    /// A scalar NumKong's CUDA kernels take: its C `nk_dtype_t`, the cuBLASLt types of the rival
    /// GEMM where cuBLASLt multiplies it, and a random operand.
    pub trait GpuScalar: Dots<Accumulator: Default + Into<f64>> {
        const DTYPE: u32;
        const CUBLASLT: Option<CublasLtTypes>;
        /// What cuBLASLt multiplies `Self` into, which is NumKong's accumulator but for F32.
        type Product: StorageElement + Default + Into<f64>;
        /// Uniform over `[-1, 1)` for floats, ±100 for `i8`, up to 200 for `u8`, any nibbles when packed.
        fn random(rng: &mut StdRng) -> Self;
    }

    impl GpuScalar for f64 {
        const DTYPE: u32 = 1 << 10;
        type Product = f64;
        const CUBLASLT: Option<CublasLtTypes> = Some(CublasLtTypes {
            input: sys::cudaDataType::CUDA_R_64F,
            output: sys::cudaDataType::CUDA_R_64F,
            compute: sys::cublasComputeType_t::CUBLAS_COMPUTE_64F,
            scale: sys::cudaDataType::CUDA_R_64F,
        });
        fn random(rng: &mut StdRng) -> Self {
            rng.random_range(-1.0..1.0)
        }
    }
    impl GpuScalar for f32 {
        const DTYPE: u32 = 1 << 11;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = into_f32(sys::cudaDataType::CUDA_R_32F);
        fn random(rng: &mut StdRng) -> Self {
            rng.random_range(-1.0..1.0)
        }
    }
    impl GpuScalar for f16 {
        const DTYPE: u32 = 1 << 12;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = into_f32(sys::cudaDataType::CUDA_R_16F);
        fn random(rng: &mut StdRng) -> Self {
            f16::from_f32(rng.random_range(-1.0..1.0))
        }
    }
    impl GpuScalar for bf16 {
        const DTYPE: u32 = 1 << 13;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = into_f32(sys::cudaDataType::CUDA_R_16BF);
        fn random(rng: &mut StdRng) -> Self {
            bf16::from_f32(rng.random_range(-1.0..1.0))
        }
    }
    impl GpuScalar for e4m3 {
        const DTYPE: u32 = 1 << 14;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = into_f32(sys::cudaDataType::CUDA_R_8F_E4M3);
        fn random(rng: &mut StdRng) -> Self {
            e4m3::from_f32(rng.random_range(-1.0..1.0))
        }
    }
    // cuBLASLt multiplies E5M2 only against E4M3
    impl GpuScalar for e5m2 {
        const DTYPE: u32 = 1 << 15;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = None;
        fn random(rng: &mut StdRng) -> Self {
            e5m2::from_f32(rng.random_range(-1.0..1.0))
        }
    }
    impl GpuScalar for e2m3 {
        const DTYPE: u32 = 1 << 18;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = None;
        fn random(rng: &mut StdRng) -> Self {
            e2m3::from_f32(rng.random_range(-1.0..1.0))
        }
    }
    impl GpuScalar for e3m2 {
        const DTYPE: u32 = 1 << 19;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = None;
        fn random(rng: &mut StdRng) -> Self {
            e3m2::from_f32(rng.random_range(-1.0..1.0))
        }
    }
    impl GpuScalar for e2m1x2 {
        const DTYPE: u32 = 1 << 24;
        type Product = f32;
        const CUBLASLT: Option<CublasLtTypes> = into_f32(sys::cudaDataType::CUDA_R_4F_E2M1);
        fn random(rng: &mut StdRng) -> Self {
            e2m1x2(rng.random())
        }
    }
    impl GpuScalar for i8 {
        const DTYPE: u32 = 1 << 2;
        type Product = i32;
        const CUBLASLT: Option<CublasLtTypes> = Some(CublasLtTypes {
            input: sys::cudaDataType::CUDA_R_8I,
            output: sys::cudaDataType::CUDA_R_32I,
            compute: sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            scale: sys::cudaDataType::CUDA_R_32I,
        });
        fn random(rng: &mut StdRng) -> Self {
            rng.random_range(-100..=100)
        }
    }
    impl GpuScalar for u8 {
        const DTYPE: u32 = 1 << 6;
        type Product = u32;
        const CUBLASLT: Option<CublasLtTypes> = None;
        fn random(rng: &mut StdRng) -> Self {
            rng.random_range(0..=200)
        }
    }
    impl GpuScalar for i4x2 {
        const DTYPE: u32 = 1 << 16;
        type Product = i32;
        const CUBLASLT: Option<CublasLtTypes> = None;
        fn random(rng: &mut StdRng) -> Self {
            i4x2(rng.random())
        }
    }
    impl GpuScalar for u4x2 {
        const DTYPE: u32 = 1 << 17;
        type Product = u32;
        const CUBLASLT: Option<CublasLtTypes> = None;
        fn random(rng: &mut StdRng) -> Self {
            u4x2(rng.random())
        }
    }

    /// One GPU a suite puts rows on: its ordinal, the capabilities it dispatches with, and a stream
    /// on it, which names the device to every kernel and is freed with it.
    pub struct GpuDevice {
        pub ordinal: usize,
        pub capabilities: Capabilities,
        pub stream: *mut c_void,
    }

    /// B packed on one GPU for its packed kernels, with the columns it holds.
    pub struct GpuPacked {
        bytes: Tensor<u8, UnifiedAllocator>,
        columns: usize,
    }

    impl GpuDevice {
        /// The suffix naming this device's rows, like `<cuda:0>`.
        pub fn label(&self) -> String {
            format!("<cuda:{}>", self.ordinal)
        }

        /// Memory the host and this device both address, where every operand of a GPU row lives.
        pub fn allocator(&self) -> UnifiedAllocator {
            UnifiedAllocator::new(self.capabilities)
        }

        /// A unified `shape` tensor of random operands drawn from `seed`.
        pub fn random<Scalar: GpuScalar>(&self, shape: &[usize], seed: u64) -> Tensor<Scalar, UnifiedAllocator> {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut tensor = Tensor::zeros_in(shape, self.allocator()).expect("Failed to allocate unified memory");
            tensor
                .as_mut_slice()
                .iter_mut()
                .for_each(|value| *value = Scalar::random(&mut rng));
            tensor
        }

        /// Waits for every launch queued on the stream, so a timed round counts finished work.
        pub fn synchronize(&self) {
            unsafe { self.capabilities.synchronize(self.stream) }.expect("Failed to synchronize the stream");
        }

        /// The kernel of `kind` for `dtype` this device runs, cast to its punned signature `Kernel`.
        unsafe fn kernel<Kernel: Copy>(&self, kind: u8, dtype: u32) -> Option<Kernel> {
            let (mut kernel, mut capability) = (None, 0);
            let mask = self.capabilities.bits();
            unsafe { nk_find_kernel_punned(kind.into(), dtype as c_int, mask, &mut kernel, &mut capability) };
            kernel.map(|kernel| unsafe { std::mem::transmute_copy(&kernel) })
        }

        /// Panics naming `kernel` and this device unless `status` is `nk_success_k`.
        fn check(&self, kernel: &str, status: c_int) {
            assert_eq!(
                status, 0,
                "NumKong's {kernel} failed on cuda:{} with status {status}",
                self.ordinal
            );
        }

        /// Queues packing `matrix`, `[columns, depth]`, for this device's packed kernels, or answers
        /// `None` when it has no packer for `Scalar`.
        pub fn pack<Scalar: GpuScalar>(&self, matrix: &Tensor<Scalar, UnifiedAllocator>) -> Option<GpuPacked> {
            let &[columns, depth] = matrix.shape() else {
                panic!("Packing needs a matrix");
            };
            let size: PackSizeKernel = unsafe { self.kernel(b'P', Scalar::DTYPE) }?;
            let pack: PackKernel = unsafe { self.kernel(b'Q', Scalar::DTYPE) }?;
            let mut bytes = 0;
            self.check("dots_pack_size", unsafe { size(columns, depth, &mut bytes) });
            let mut packed =
                Tensor::<u8, _>::zeros_in(&[bytes], self.allocator()).expect("Failed to allocate unified memory");
            let stride = matrix.stride_bytes(0) as usize;
            let (source, target) = (matrix.as_ptr().cast(), packed.as_mut_ptr().cast());
            self.check("dots_pack", unsafe {
                pack(source, columns, depth, stride, target, 0, columns, self.stream)
            });
            Some(GpuPacked { bytes: packed, columns })
        }

        /// Queues `output = metric(rows, packed)`, `[rows, columns]` of the kernel's output type, or
        /// answers `None` when this device has no such kernel for `Scalar`.
        pub fn packed<Scalar: GpuScalar, Output: StorageElement>(
            &self,
            metric: GpuMetric,
            rows: &Tensor<Scalar, UnifiedAllocator>,
            packed: &GpuPacked,
            output: &mut Tensor<Output, UnifiedAllocator>,
        ) -> Option<()> {
            let kernel: PackedKernel = unsafe { self.kernel(metric.kinds().0, Scalar::DTYPE) }?;
            let &[count, depth] = rows.shape() else {
                panic!("The packed kernels need a matrix of rows");
            };
            let strides = (rows.stride_bytes(0) as usize, output.stride_bytes(0) as usize);
            let (source, target) = (rows.as_ptr().cast(), output.as_mut_ptr().cast());
            let status = unsafe {
                kernel(
                    source,
                    packed.bytes.as_ptr().cast(),
                    target,
                    count,
                    packed.columns,
                    depth,
                    strides.0,
                    strides.1,
                    self.stream,
                )
            };
            self.check("packed", status);
            Some(())
        }

        /// Queues the upper triangle of `output = metric(vectors, vectors)`, or answers `None` when
        /// this device has no such kernel for `Scalar`.
        pub fn symmetric<Scalar: GpuScalar, Output: StorageElement>(
            &self,
            metric: GpuMetric,
            vectors: &Tensor<Scalar, UnifiedAllocator>,
            output: &mut Tensor<Output, UnifiedAllocator>,
        ) -> Option<()> {
            let kernel: SymmetricKernel = unsafe { self.kernel(metric.kinds().1, Scalar::DTYPE) }?;
            let &[count, depth] = vectors.shape() else {
                panic!("The symmetric kernels need a matrix of vectors");
            };
            let strides = (vectors.stride_bytes(0) as usize, output.stride_bytes(0) as usize);
            let (source, target) = (vectors.as_ptr().cast(), output.as_mut_ptr().cast());
            let status = unsafe {
                kernel(
                    source,
                    count,
                    depth,
                    strides.0,
                    target,
                    strides.1,
                    0,
                    count,
                    self.stream,
                )
            };
            self.check("symmetric", status);
            Some(())
        }

        /// Packs `keys` and `values`, `[tokens, heads × depth]` of one segment, for this device's
        /// attention kernel, or answers `None` when it has none for `Scalar`.
        pub fn attention_pack<Scalar: GpuScalar>(
            &self,
            keys: &Tensor<Scalar, UnifiedAllocator>,
            values: &Tensor<Scalar, UnifiedAllocator>,
            depth: usize,
        ) -> Option<GpuKeyValues> {
            let &[tokens, width] = keys.shape() else {
                panic!("Packing needs a matrix of keys");
            };
            let heads = width / depth;
            let size: AttentionPackSizeKernel = unsafe { self.kernel(b'B', Scalar::DTYPE) }?;
            let pack: AttentionPackKernel = unsafe { self.kernel(b'V', Scalar::DTYPE) }?;
            let mut bytes = 0;
            self.check("attention_pack_size", unsafe {
                size(heads, depth, tokens, 1, &mut bytes)
            });
            let mut packed = Tensor::<u8, _>::zeros_in(&[bytes], self.allocator()).expect("Failed to allocate");
            let offsets = Tensor::<u32, _>::from_slice_in(&[0, tokens as u32], &[2], self.allocator())
                .expect("Failed to allocate");
            let strides = (keys.stride_bytes(0) as usize, values.stride_bytes(0) as usize);
            let (keys, values, target) = (keys.as_ptr().cast(), values.as_ptr().cast(), packed.as_mut_ptr().cast());
            let status = unsafe {
                pack(
                    heads,
                    depth,
                    offsets.as_ptr(),
                    std::ptr::null(),
                    1,
                    keys,
                    strides.0,
                    values,
                    strides.1,
                    target,
                    0,
                    heads,
                    self.stream,
                )
            };
            self.check("attention_pack", status);
            // The packing reads the offsets as it runs, and they are freed on return.
            self.synchronize();
            Some(GpuKeyValues {
                bytes: packed,
                heads,
                depth,
            })
        }

        /// Queues bidirectional attention of `queries`, `[tokens, heads × depth]` of one segment
        /// spanning `query_offsets`, over `key_values` into F32 `output`, or answers `None` when this
        /// device has no attention kernel for `Scalar`.
        pub fn attention<Scalar: GpuScalar>(
            &self,
            queries: &Tensor<Scalar, UnifiedAllocator>,
            key_values: &GpuKeyValues,
            query_offsets: &Tensor<u32, UnifiedAllocator>,
            output: &mut Tensor<f32, UnifiedAllocator>,
        ) -> Option<()> {
            let kernel: AttentionPackedKernel = unsafe { self.kernel(b'F', Scalar::DTYPE) }?;
            let (tokens, heads, depth) = (queries.shape()[0], key_values.heads, key_values.depth);
            let scale = 1.0 / (depth as f32).sqrt();
            let strides = (queries.stride_bytes(0) as usize, output.stride_bytes(0) as usize);
            let (source, target) = (queries.as_ptr().cast(), output.as_mut_ptr().cast());
            let status = unsafe {
                kernel(
                    heads,
                    heads,
                    depth,
                    query_offsets.as_ptr(),
                    tokens,
                    scale,
                    usize::MAX,
                    usize::MAX,
                    source,
                    strides.0,
                    key_values.bytes.as_ptr().cast(),
                    target,
                    strides.1,
                    std::ptr::null_mut(),
                    0,
                    tokens * heads,
                    self.stream,
                )
            };
            self.check("attention_packed", status);
            Some(())
        }
    }

    /// Keys and values packed on one GPU for its attention kernel, with their heads and depth.
    pub struct GpuKeyValues {
        bytes: Tensor<u8, UnifiedAllocator>,
        heads: usize,
        depth: usize,
    }

    impl Drop for GpuDevice {
        fn drop(&mut self) {
            self.synchronize();
            unsafe { Capabilities::stream_free_cuda(self.stream) }.expect("Failed to free the stream");
        }
    }

    /// The GPUs that carry rows: each one `NUMWARS_DEVICES` names, refusing the run when one cannot
    /// be opened, or else device zero, skipped on a machine without one.
    pub fn gpu_devices(settings: &Settings) -> Vec<GpuDevice> {
        let mut devices = Vec::new();
        for ordinal in settings.devices.clone().unwrap_or_else(|| vec![0]) {
            let opened = Capabilities::cuda_enabled(ordinal).and_then(|capabilities| {
                let stream = unsafe { Capabilities::stream_init_cuda(ordinal) }?;
                Ok(GpuDevice {
                    ordinal,
                    capabilities,
                    stream,
                })
            });
            match opened {
                Ok(device) => {
                    println!("- GPU device: cuda:{ordinal} running {}", device.capabilities);
                    devices.push(device);
                }
                Err(error) if settings.devices.is_some() => {
                    eprintln!("Requested CUDA device cuda:{ordinal} is unavailable: {error}");
                    std::process::exit(1);
                }
                Err(_) => {}
            }
        }
        devices
    }

    /// Whether the first round of `name`, a `[rows, columns]` result, matches the CPU's `expected`
    /// leading rows within `tolerance` of each expected value, or of their root mean square if
    /// larger, as a sum near zero carries the rounding of its large terms, naming the first mismatch
    /// if not. A symmetric result is compared above its diagonal only, where a vector's distance to
    /// itself cancels to rounding noise.
    pub fn gpu_check<Found: Copy + Into<f64>, Expected: Copy + Into<f64>>(
        name: &str,
        found: &[Found],
        expected: &[Expected],
        columns: usize,
        upper_only: bool,
        tolerance: f64,
    ) -> Result<(), String> {
        let squares: f64 = expected.iter().map(|&value| Into::<f64>::into(value).powi(2)).sum();
        let root_mean_square = (squares / expected.len() as f64).sqrt();
        for (row, expected_row) in expected.chunks(columns).enumerate() {
            for column in (if upper_only { row + 1 } else { 0 })..columns {
                let (found, expected) = (found[row * columns + column].into(), expected_row[column].into());
                if !((found - expected).abs() <= tolerance * expected.abs().max(root_mean_square)) {
                    return Err(format!(
                        "{name} disagrees with the CPU on its first round at [{row}, {column}]: {found} against {expected}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// One cuBLASLt `C = A × Bᵀ` over row-major operands in unified memory, planned once with the
    /// algorithm cuBLASLt ranks first, and queued on a device's stream.
    pub struct CublasLtGemm {
        handle: sys::cublasLtHandle_t,
        description: sys::cublasLtMatmulDesc_t,
        layouts: [sys::cublasLtMatrixLayout_t; 3],
        algorithm: sys::cublasLtMatmulAlgo_t,
        workspace: Tensor<u8, UnifiedAllocator>,
        scales: Vec<Tensor<u8, UnifiedAllocator>>,
        alpha: [u8; 8],
        beta: [u8; 8],
        stream: *mut c_void,
    }

    /// The row-major operands of one cuBLASLt GEMM, `C = A × Bᵀ`, or `C = A × B` when B is
    /// `[depth, columns]`, repeated over `batches` with each operand's start moving by its batch
    /// stride. Strides count elements, ordered A, B, C.
    #[derive(Clone, Copy)]
    pub struct CublasLtLayout {
        pub rows: usize,
        pub columns: usize,
        pub depth: usize,
        pub b_transposed: bool,
        pub row_strides: [usize; 3],
        pub batches: usize,
        pub batch_strides: [usize; 3],
    }

    impl CublasLtLayout {
        /// One `[rows, depth] × [columns, depth]ᵀ` over contiguous rows.
        pub fn new(rows: usize, columns: usize, depth: usize) -> Self {
            CublasLtLayout {
                rows,
                columns,
                depth,
                b_transposed: true,
                row_strides: [depth, depth, columns],
                batches: 1,
                batch_strides: [0; 3],
            }
        }
    }

    impl CublasLtGemm {
        /// Plans the GEMM `layout` describes on `device`, or answers the error cuBLASLt gives for
        /// types or a shape it has no kernel for.
        pub fn new(
            device: &GpuDevice,
            types: CublasLtTypes,
            layout: CublasLtLayout,
        ) -> Result<Self, cublaslt::CublasError> {
            use sys::cublasLtMatmulDescAttributes_t::*;
            let CublasLtLayout {
                rows, columns, depth, ..
            } = layout;
            let mut gemm = CublasLtGemm {
                handle: cublaslt::create_handle()?,
                description: std::ptr::null_mut(),
                layouts: [std::ptr::null_mut(); 3],
                algorithm: unsafe { std::mem::zeroed() },
                workspace: Tensor::zeros_in(&[WORKSPACE_BYTES], device.allocator()).expect("Failed to allocate"),
                scales: Vec::new(),
                alpha: [0; 8],
                beta: [0; 8],
                stream: device.stream,
            };
            let description = cublaslt::create_matmul_desc(types.compute, types.scale)?;
            gemm.description = description;
            let set = |attribute, value: *const c_void, bytes| unsafe {
                cublaslt::set_matmul_desc_attribute(description, attribute, value, bytes)
            };
            // Row-major `A × Bᵀ` is column-major `B × Aᵀ` with B transposed, the TN layout FP8 and INT8 need.
            let first_operation = match layout.b_transposed {
                true => cublasOperation_t::CUBLAS_OP_T,
                false => cublasOperation_t::CUBLAS_OP_N,
            };
            let operations = [first_operation, cublasOperation_t::CUBLAS_OP_N];
            set(
                CUBLASLT_MATMUL_DESC_TRANSA,
                (&operations[0] as *const cublasOperation_t).cast(),
                4,
            )?;
            set(
                CUBLASLT_MATMUL_DESC_TRANSB,
                (&operations[1] as *const cublasOperation_t).cast(),
                4,
            )?;
            if types.input == sys::cudaDataType::CUDA_R_4F_E2M1 {
                // FP4 takes a UE4M3 scale per 16 values, here all 1.0, so the tiled layout cannot matter.
                let mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 as u32;
                let operands = [
                    (
                        columns,
                        CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    ),
                    (
                        rows,
                        CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    ),
                ];
                for (outer, mode_attribute, pointer_attribute) in operands {
                    let bytes = outer.next_multiple_of(128) * (depth / 16).next_multiple_of(4);
                    let ones = Tensor::full_in(&[bytes], 0x38u8, device.allocator()).expect("Failed to allocate");
                    let pointer = ones.as_ptr();
                    set(mode_attribute, (&mode as *const u32).cast(), 4)?;
                    set(
                        pointer_attribute,
                        (&pointer as *const *const u8).cast(),
                        size_of::<*const u8>(),
                    )?;
                    gemm.scales.push(ones);
                }
            }
            let (depth, columns, rows) = (depth as u64, columns as u64, rows as u64);
            let [a_stride, b_stride, c_stride] = layout.row_strides.map(|stride| stride as i64);
            let (b_rows, b_columns) = if layout.b_transposed {
                (depth, columns)
            } else {
                (columns, depth)
            };
            gemm.layouts[0] = cublaslt::create_matrix_layout(types.input, b_rows, b_columns, b_stride)?;
            gemm.layouts[1] = cublaslt::create_matrix_layout(types.input, depth, rows, a_stride)?;
            gemm.layouts[2] = cublaslt::create_matrix_layout(types.output, columns, rows, c_stride)?;
            if layout.batches > 1 {
                use sys::cublasLtMatrixLayoutAttribute_t::*;
                let count = layout.batches as i32;
                let [a_batch, b_batch, c_batch] = layout.batch_strides.map(|stride| stride as i64);
                for (matrix, batch_stride) in gemm.layouts.into_iter().zip([b_batch, a_batch, c_batch]) {
                    unsafe {
                        let count_pointer = (&count as *const i32).cast();
                        cublaslt::set_matrix_layout_attribute(
                            matrix,
                            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                            count_pointer,
                            4,
                        )?;
                        let offset_pointer = (&batch_stride as *const i64).cast();
                        let offset = CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET;
                        cublaslt::set_matrix_layout_attribute(matrix, offset, offset_pointer, 8)?;
                    }
                }
            }
            let preference = cublaslt::create_matmul_pref()?;
            let [first, second, output] = gemm.layouts;
            let limit = WORKSPACE_BYTES as u64;
            let attribute = sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
            let heuristic = unsafe {
                cublaslt::set_matmul_pref_attribute(preference, attribute, (&limit as *const u64).cast(), 8).and_then(
                    |()| {
                        cublaslt::get_matmul_algo_heuristic(
                            gemm.handle,
                            description,
                            first,
                            second,
                            output,
                            output,
                            preference,
                        )
                    },
                )
            };
            unsafe { cublaslt::destroy_matmul_pref(preference) }?;
            gemm.algorithm = heuristic?.algo;
            (gemm.alpha, gemm.beta) = match types.scale {
                sys::cudaDataType::CUDA_R_64F => (1f64.to_ne_bytes(), 0f64.to_ne_bytes()),
                sys::cudaDataType::CUDA_R_32I => ((1u64).to_ne_bytes(), 0u64.to_ne_bytes()),
                _ => ((1f32.to_bits() as u64).to_ne_bytes(), 0u64.to_ne_bytes()),
            };
            Ok(gemm)
        }

        /// Queues `c = a × bᵀ`, or `a × b`, as planned, on the stream it was planned for.
        pub fn launch<Input: StorageElement, Output: StorageElement>(
            &self,
            a: &Tensor<Input, UnifiedAllocator>,
            b: &Tensor<Input, UnifiedAllocator>,
            c: &mut Tensor<Output, UnifiedAllocator>,
        ) {
            let [first, second, output] = self.layouts;
            let (alpha, beta) = (self.alpha.as_ptr().cast(), self.beta.as_ptr().cast());
            let workspace = self.workspace.as_ptr() as *mut c_void;
            let target = c.as_mut_ptr().cast::<c_void>();
            unsafe {
                cublaslt::matmul(
                    self.handle,
                    self.description,
                    alpha,
                    beta,
                    b.as_ptr().cast(),
                    first,
                    a.as_ptr().cast(),
                    second,
                    target,
                    output,
                    target,
                    output,
                    &self.algorithm,
                    workspace,
                    WORKSPACE_BYTES,
                    self.stream.cast(),
                )
            }
            .expect("cuBLASLt failed to queue its GEMM");
        }
    }

    impl Drop for CublasLtGemm {
        fn drop(&mut self) {
            unsafe {
                for layout in self.layouts.into_iter().filter(|layout| !layout.is_null()) {
                    let _ = cublaslt::destroy_matrix_layout(layout);
                }
                if !self.description.is_null() {
                    let _ = cublaslt::destroy_matmul_desc(self.description);
                }
                let _ = cublaslt::destroy_handle(self.handle);
            }
        }
    }
}

// endregion
