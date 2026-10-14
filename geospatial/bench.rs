//! Benchmark for many-to-many geospatial distances
//!
//! Scores every row point against every column point into a row-major matrix, comparing NumKong's
//! packed kernels against the `geo` crate and scalar baselines. Each point set packs once, outside
//! the timed loop, as the dots suite packs its B operand.
//!
//! Competitors:
//! - numkong (HaversinesPackedMatrix, VincentysPackedMatrix, SqEuclideans3D)
//! - geo (Haversine on the same mean sphere, Karney's Geodesic on the same WGS-84 ellipsoid)
//! - baseline (scalar loops over degrees, like the `geo` rows)
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_geospatial --bench bench_geospatial
//! NUMWARS_FILTER="haversines" cargo bench --features bench_geospatial
//! ```
//!
//! Benchmark naming: geospatial/{metric}/{dtype}/{rows}x{columns}
//! Examples: geospatial/haversines/f64/1024x128, geospatial/sqeuclideans3d/i32/1024x128

use std::hint::black_box;
use std::ops::Neg;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput};
use geo::{Distance, Geodesic, Haversine as GeoHaversine, Point as GeoPoint};
use num_traits::Float;
use numkong::{Capabilities, HaversinesPackedMatrix, SqEuclideans3D, VincentysPackedMatrix};
use rand::distr::uniform::SampleUniform;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use numwars::Settings;

/// The IUGG mean Earth radius, which NumKong's and `geo`'s Haversine both use.
const EARTH_RADIUS_METERS: f64 = 6_371_008.8;

// region: Baseline Implementations

fn baseline_haversine<T: Float>(latitude_a: T, longitude_a: T, latitude_b: T, longitude_b: T) -> T {
    let earth_radius = T::from(EARTH_RADIUS_METERS).unwrap();
    let two = T::from(2.0).unwrap();
    let delta_latitude = (latitude_b - latitude_a).to_radians();
    let delta_longitude = (longitude_b - longitude_a).to_radians();
    let half_chord_squared = (delta_latitude / two).sin().powi(2)
        + latitude_a.to_radians().cos() * latitude_b.to_radians().cos() * (delta_longitude / two).sin().powi(2);
    earth_radius * two * half_chord_squared.sqrt().asin()
}

/// Baseline Vincenty inverse formula (iterative, WGS-84 ellipsoid).
///
/// Computes the geodesic distance between two points on the WGS-84 ellipsoid
/// using Vincenty's iterative method. Returns distance in meters.
fn baseline_vincenty<T: Float>(lat1: T, lon1: T, lat2: T, lon2: T) -> T {
    let a = T::from(6_378_137.0).unwrap(); // semi-major axis (meters)
    let f = T::from(1.0 / 298.257_223_563).unwrap(); // flattening
    let b = a * (T::one() - f); // semi-minor axis

    let phi1 = lat1.to_radians();
    let phi2 = lat2.to_radians();
    let l = (lon2 - lon1).to_radians(); // difference in longitude

    // Reduced latitudes
    let u1 = ((T::one() - f) * phi1.tan()).atan();
    let u2 = ((T::one() - f) * phi2.tan()).atan();
    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    // Iterative computation
    let mut lambda = l;
    let max_iterations = 200;
    let tolerance = T::from(1e-12).unwrap();
    let two = T::from(2.0).unwrap();

    let mut sin_sigma = T::zero();
    let mut cos_sigma = T::zero();
    let mut sigma = T::zero();
    let mut sin_alpha;
    let mut cos_sq_alpha = T::zero();
    let mut cos_2sigma_m = T::zero();

    for _ in 0..max_iterations {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        sin_sigma = ((cos_u2 * sin_lambda).powi(2) + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2)).sqrt();

        // Co-incident points
        if sin_sigma == T::zero() {
            return T::zero();
        }

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = sin_sigma.atan2(cos_sigma);

        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        cos_sq_alpha = T::one() - sin_alpha * sin_alpha;

        cos_2sigma_m = if cos_sq_alpha != T::zero() {
            cos_sigma - two * sin_u1 * sin_u2 / cos_sq_alpha
        } else {
            T::zero() // equatorial line
        };

        let c = f / T::from(16.0).unwrap()
            * cos_sq_alpha
            * (T::from(4.0).unwrap() + f * (T::from(4.0).unwrap() - T::from(3.0).unwrap() * cos_sq_alpha));

        let lambda_prev = lambda;
        lambda = l
            + (T::one() - c)
                * f
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (T::from(-1.0).unwrap() + two * cos_2sigma_m * cos_2sigma_m)));

        if (lambda - lambda_prev).abs() < tolerance {
            break;
        }
    }

    // Calculate distance
    let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);
    let cap_a = T::one()
        + u_sq / T::from(16384.0).unwrap()
            * (T::from(4096.0).unwrap()
                + u_sq
                    * (T::from(-768.0).unwrap() + u_sq * (T::from(320.0).unwrap() - T::from(175.0).unwrap() * u_sq)));
    let cap_b = u_sq / T::from(1024.0).unwrap()
        * (T::from(256.0).unwrap()
            + u_sq * (T::from(-128.0).unwrap() + u_sq * (T::from(74.0).unwrap() - T::from(47.0).unwrap() * u_sq)));

    let delta_sigma = cap_b
        * sin_sigma
        * (cos_2sigma_m
            + cap_b / T::from(4.0).unwrap()
                * (cos_sigma * (T::from(-1.0).unwrap() + two * cos_2sigma_m * cos_2sigma_m)
                    - cap_b / T::from(6.0).unwrap()
                        * cos_2sigma_m
                        * (T::from(-3.0).unwrap() + T::from(4.0).unwrap() * sin_sigma * sin_sigma)
                        * (T::from(-3.0).unwrap() + T::from(4.0).unwrap() * cos_2sigma_m * cos_2sigma_m)));

    b * cap_a * (sigma - delta_sigma)
}

/// Squared distance between two Earth-fixed points, exact in `u64` for `i32` centimeters.
trait BaselineSqEuclidean3D: SqEuclideans3D + Copy {
    fn baseline_sqeuclidean3d(a: [Self; 3], b: [Self; 3]) -> Self::Distance;
}

impl BaselineSqEuclidean3D for i32 {
    fn baseline_sqeuclidean3d(a: [i32; 3], b: [i32; 3]) -> u64 {
        let square = |axis: usize| (a[axis] as i64 - b[axis] as i64).unsigned_abs().pow(2);
        square(0) + square(1) + square(2)
    }
}

impl BaselineSqEuclidean3D for f64 {
    fn baseline_sqeuclidean3d(a: [f64; 3], b: [f64; 3]) -> f64 {
        let square = |axis: usize| (a[axis] - b[axis]).powi(2);
        square(0) + square(1) + square(2)
    }
}

// endregion

// region: Data Generation

fn generate_random_coords<T: SampleUniform + PartialOrd + Copy>(
    rng: &mut impl Rng,
    count: usize,
    min: T,
    max: T,
) -> Vec<T> {
    (0..count).map(|_| rng.random_range(min..max)).collect()
}

/// Latitudes and longitudes in degrees, as `geo` and the baselines take them.
struct Coordinates {
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
}

impl Coordinates {
    fn random(rng: &mut impl Rng, count: usize) -> Self {
        Coordinates {
            latitudes: generate_random_coords(rng, count, -90.0, 90.0),
            longitudes: generate_random_coords(rng, count, -180.0, 180.0),
        }
    }

    /// Latitudes and longitudes in radians, as NumKong packs them.
    fn radians(&self) -> (Vec<f64>, Vec<f64>) {
        let radians = |degrees: &[f64]| degrees.iter().map(|degree| degree.to_radians()).collect();
        (radians(&self.latitudes), radians(&self.longitudes))
    }

    fn pairs(&self) -> Vec<(f64, f64)> {
        self.latitudes
            .iter()
            .copied()
            .zip(self.longitudes.iter().copied())
            .collect()
    }

    fn geo_points(&self) -> Vec<GeoPoint<f64>> {
        let points = self.latitudes.iter().zip(&self.longitudes);
        points
            .map(|(&latitude, &longitude)| GeoPoint::new(longitude, latitude))
            .collect()
    }
}

/// Earth-fixed points within `radius` of the center on every axis.
fn generate_random_points<T: SampleUniform + PartialOrd + Neg<Output = T> + Copy>(
    rng: &mut impl Rng,
    count: usize,
    radius: T,
) -> Vec<[T; 3]> {
    (0..count)
        .map(|_| std::array::from_fn(|_| rng.random_range(-radius..radius)))
        .collect()
}

// endregion

// region: Per-Library Runs

/// Times `distance` from every row to every column into a row-major __[rows,columns]__ matrix.
fn run_matrix<Point: Copy, Output: Copy + Default>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    library: &str,
    rows: &[Point],
    columns: &[Point],
    distance: impl Fn(Point, Point) -> Output,
) {
    let mut results = vec![Output::default(); rows.len() * columns.len()];
    group.bench_function(library, |bench| {
        bench.iter(|| {
            for (row_results, &row) in results.chunks_exact_mut(columns.len()).zip(rows) {
                for (result, &column) in row_results.iter_mut().zip(columns) {
                    *result = distance(row, column);
                }
            }
            black_box(&results);
        })
    });
}

// endregion

// region: Benchmarks

/// Opens the group `geospatial/{metric}/{dtype}/{rows}x{columns}`, or `None` when it is filtered out.
fn open_group<'a>(
    c: &'a mut Criterion,
    settings: &Settings,
    metric: &str,
    dtype: &str,
) -> Option<BenchmarkGroup<'a, WallTime>> {
    let (rows, columns) = (settings.dims_height, settings.dims_width);
    let name = format!("geospatial/{metric}/{dtype}/{rows}x{columns}");
    if !settings.selects(&name) {
        return None;
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements((rows * columns) as u64));
    Some(group)
}

/// Benchmark great-circle distances on the IUGG mean sphere.
pub fn bench_haversines(c: &mut Criterion, settings: &Settings) {
    let Some(mut group) = open_group(c, settings, "haversines", "f64") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(settings.seed.into());
    let rows = Coordinates::random(&mut rng, settings.dims_height);
    let columns = Coordinates::random(&mut rng, settings.dims_width);

    let ((row_latitudes, row_longitudes), (column_latitudes, column_longitudes)) = (rows.radians(), columns.radians());
    let row_pack = HaversinesPackedMatrix::new(&row_latitudes, &row_longitudes).expect("Failed to pack rows");
    let column_pack =
        HaversinesPackedMatrix::new(&column_latitudes, &column_longitudes).expect("Failed to pack columns");
    let mut meters = vec![0.0f64; settings.dims_height * settings.dims_width];
    group.bench_function("numkong", |bench| {
        bench.iter(|| {
            row_pack
                .haversines(&column_pack, &mut meters)
                .expect("Haversine failed");
            black_box(&meters);
        })
    });
    run_matrix(
        &mut group,
        "baseline",
        &rows.pairs(),
        &columns.pairs(),
        |(latitude_a, longitude_a), (latitude_b, longitude_b)| {
            baseline_haversine(latitude_a, longitude_a, latitude_b, longitude_b)
        },
    );
    run_matrix(&mut group, "geo", &rows.geo_points(), &columns.geo_points(), |a, b| {
        GeoHaversine.distance(a, b)
    });
    group.finish();
}

/// Benchmark geodesic distances on the WGS-84 ellipsoid.
pub fn bench_vincentys(c: &mut Criterion, settings: &Settings) {
    let Some(mut group) = open_group(c, settings, "vincentys", "f64") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(settings.seed.into());
    let rows = Coordinates::random(&mut rng, settings.dims_height);
    let columns = Coordinates::random(&mut rng, settings.dims_width);

    let ((row_latitudes, row_longitudes), (column_latitudes, column_longitudes)) = (rows.radians(), columns.radians());
    let row_pack = VincentysPackedMatrix::new(&row_latitudes, &row_longitudes).expect("Failed to pack rows");
    let column_pack =
        VincentysPackedMatrix::new(&column_latitudes, &column_longitudes).expect("Failed to pack columns");
    let mut meters = vec![0.0f64; settings.dims_height * settings.dims_width];
    group.bench_function("numkong", |bench| {
        bench.iter(|| {
            row_pack.vincentys(&column_pack, &mut meters).expect("Vincenty failed");
            black_box(&meters);
        })
    });
    run_matrix(
        &mut group,
        "baseline",
        &rows.pairs(),
        &columns.pairs(),
        |(latitude_a, longitude_a), (latitude_b, longitude_b)| {
            baseline_vincenty(latitude_a, longitude_a, latitude_b, longitude_b)
        },
    );
    run_matrix(&mut group, "geo", &rows.geo_points(), &columns.geo_points(), |a, b| {
        Geodesic.distance(a, b)
    });
    group.finish();
}

fn bench_sqeuclideans3d_dtype<T>(c: &mut Criterion, settings: &Settings, dtype: &str, radius: T)
where
    T: BaselineSqEuclidean3D + SampleUniform + PartialOrd + Neg<Output = T>,
{
    let Some(mut group) = open_group(c, settings, "sqeuclideans3d", dtype) else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(settings.seed.into());
    let rows = generate_random_points(&mut rng, settings.dims_height, radius);
    let columns = generate_random_points(&mut rng, settings.dims_width, radius);

    let mut squares = vec![T::Distance::default(); rows.len() * columns.len()];
    group.bench_function("numkong", |bench| {
        bench.iter(|| {
            T::sqeuclideans3d(&rows, &columns, &mut squares).expect("SqEuclideans3D failed");
            black_box(&squares);
        })
    });
    run_matrix(&mut group, "baseline", &rows, &columns, T::baseline_sqeuclidean3d);
    group.finish();
}

/// Benchmark squared distances between Earth-fixed points: `i32` centimeters and `f64` meters.
pub fn bench_sqeuclideans3d(c: &mut Criterion, settings: &Settings) {
    bench_sqeuclideans3d_dtype(c, settings, "i32", 637_100_000i32);
    bench_sqeuclideans3d_dtype(c, settings, "f64", 6_371_000.0f64);
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn haversine_baseline_correctness() {
        // New York to London: ~5570 km
        let distance = baseline_haversine(40.7128f64, -74.0060, 51.5074, -0.1278);
        assert!((distance - 5_570_000.0).abs() < 50_000.0);
    }

    #[test]
    fn haversine_baseline_same_point() {
        let distance = baseline_haversine(0.0f64, 0.0, 0.0, 0.0);
        assert!(distance.abs() < 1e-6);
    }

    #[test]
    fn vincenty_baseline_correctness() {
        // New York to London: ~5570 km (Vincenty is more precise on the ellipsoid)
        let distance = baseline_vincenty(40.7128f64, -74.0060, 51.5074, -0.1278);
        assert!(
            (distance - 5_570_000.0).abs() < 50_000.0,
            "Expected ~5570 km, got {} m",
            distance
        );
    }

    #[test]
    fn vincenty_baseline_same_point() {
        let distance = baseline_vincenty(0.0f64, 0.0, 0.0, 0.0);
        assert!(distance.abs() < 1e-6);
    }

    #[test]
    fn sqeuclidean3d_baseline_exact() {
        let far = 637_100_000i32;
        assert_eq!(
            i32::baseline_sqeuclidean3d([-far, 0, 0], [far, 0, 0]),
            (2 * far as u64).pow(2)
        );
        assert_eq!(f64::baseline_sqeuclidean3d([0.0, 3.0, 4.0], [0.0, 0.0, 0.0]), 25.0);
    }
}

// endregion

// region: Main

fn main() {
    let settings = Settings::read();
    Capabilities::cpu_enabled()
        .configure_thread()
        .expect("Failed to configure the thread");
    let mut criterion = numwars::configure_criterion(&settings);
    bench_haversines(&mut criterion, &settings);
    bench_vincentys(&mut criterion, &settings);
    bench_sqeuclideans3d(&mut criterion, &settings);
    criterion.final_summary();
}

// endregion
