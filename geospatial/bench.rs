//! Benchmark for geospatial distance operations
//!
//! Compares NumKong vs baseline and geo crate for geodesic distance metrics.
//!
//! Run with:
//! ```bash
//! cargo bench --features bench_geospatial --bench bench_geospatial
//! NUMWARS_FILTER="haversine" cargo bench --features bench_geospatial
//! ```
//!
//! Environment variables:
//! - NUMWARS_DIMS: Number of coordinate pairs (default: 1536)
//! - NUMWARS_FILTER: Regex to filter benchmark names
//!
//! Benchmark naming: geospatial/{metric}/{dtype}
//! Examples: geospatial/haversine/f32, geospatial/haversine/f64

#[path = "../utils.rs"]
mod utils;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use geo::{point, Distance, Geodesic, Haversine as GeoHaversine};
use num_traits::Float;
use numkong::{Haversine, Vincenty};
use rand::distr::uniform::SampleUniform;
use rand::{Rng, RngExt};
use std::hint::black_box;
use utils::*;

// region: Baseline Implementations

fn baseline_haversine<T: Float>(latitude_a: T, longitude_a: T, latitude_b: T, longitude_b: T) -> T {
    let earth_radius_km = T::from(6371.0).unwrap();
    let two = T::from(2.0).unwrap();
    let delta_latitude = (latitude_b - latitude_a).to_radians();
    let delta_longitude = (longitude_b - longitude_a).to_radians();
    let half_chord_squared = (delta_latitude / two).sin().powi(2)
        + latitude_a.to_radians().cos() * latitude_b.to_radians().cos() * (delta_longitude / two).sin().powi(2);
    earth_radius_km * two * half_chord_squared.sqrt().asin()
}

/// Baseline Vincenty inverse formula (iterative, WGS-84 ellipsoid).
///
/// Computes the geodesic distance between two points on the WGS-84 ellipsoid
/// using Vincenty's iterative method. Returns distance in kilometers.
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
    let mut sin_alpha = T::zero();
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

    let distance_meters = b * cap_a * (sigma - delta_sigma);

    // Convert meters to kilometers
    distance_meters / T::from(1000.0).unwrap()
}

fn baseline_haversine_batch<T: Float>(
    a_lats: &[T],
    a_lons: &[T],
    b_lats: &[T],
    b_lons: &[T],
    result: &mut [T],
) -> Option<()> {
    let n = a_lats.len();
    if a_lons.len() != n || b_lats.len() != n || b_lons.len() != n || result.len() != n {
        return None;
    }
    for i in 0..n {
        result[i] = baseline_haversine(a_lats[i], a_lons[i], b_lats[i], b_lons[i]);
    }
    Some(())
}

fn baseline_vincenty_batch<T: Float>(
    a_lats: &[T],
    a_lons: &[T],
    b_lats: &[T],
    b_lons: &[T],
    result: &mut [T],
) -> Option<()> {
    let n = a_lats.len();
    if a_lons.len() != n || b_lats.len() != n || b_lons.len() != n || result.len() != n {
        return None;
    }
    for i in 0..n {
        result[i] = baseline_vincenty(a_lats[i], a_lons[i], b_lats[i], b_lons[i]);
    }
    Some(())
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

trait GeoScalar: Float + SampleUniform + Copy + 'static {
    fn to_f64(self) -> f64;
}

impl GeoScalar for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl GeoScalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

fn to_geo_points<T: GeoScalar>(lats: &[T], lons: &[T]) -> Vec<geo::Point> {
    lats.iter()
        .zip(lons)
        .map(|(&lat, &lon)| point!(x: lon.to_f64(), y: lat.to_f64()))
        .collect()
}

// endregion

// region: Per-library Run traits

trait RunBaselineHaversine: Float + Sized {
    fn run(
        _g: &mut BenchmarkGroup<'_, WallTime>,
        _a_lats: &[Self],
        _a_lons: &[Self],
        _b_lats: &[Self],
        _b_lons: &[Self],
    ) {
    }
}

impl<T: Float> RunBaselineHaversine for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a_lats: &[T], a_lons: &[T], b_lats: &[T], b_lons: &[T]) {
        let mut results = vec![T::zero(); a_lats.len()];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                black_box(baseline_haversine_batch(a_lats, a_lons, b_lats, b_lons, &mut results));
                black_box(&results);
            })
        });
    }
}

trait RunNumKongHaversine: Haversine + Float + Sized {
    fn run(
        _g: &mut BenchmarkGroup<'_, WallTime>,
        _a_lats: &[Self],
        _a_lons: &[Self],
        _b_lats: &[Self],
        _b_lons: &[Self],
    ) {
    }
}

impl<T: Haversine + Float> RunNumKongHaversine for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a_lats: &[T], a_lons: &[T], b_lats: &[T], b_lons: &[T]) {
        let mut results = vec![T::zero(); a_lats.len()];
        group.bench_function("numkong", |bench| {
            bench.iter(|| {
                black_box(Haversine::haversine(a_lats, a_lons, b_lats, b_lons, &mut results));
                black_box(&results);
            })
        });
    }
}

trait RunGeoHaversine: Sized {
    fn run(
        _g: &mut BenchmarkGroup<'_, WallTime>,
        _a_lats: &[Self],
        _a_lons: &[Self],
        _b_lats: &[Self],
        _b_lons: &[Self],
    ) {
    }
}

impl<T: GeoScalar> RunGeoHaversine for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a_lats: &[T], a_lons: &[T], b_lats: &[T], b_lons: &[T]) {
        let points_a = to_geo_points(a_lats, a_lons);
        let points_b = to_geo_points(b_lats, b_lons);
        let mut results = vec![0.0f64; points_a.len()];
        group.bench_function("geo", |bench| {
            bench.iter(|| {
                for (i, (a, b)) in points_a.iter().zip(&points_b).enumerate() {
                    results[i] = GeoHaversine.distance(*a, *b);
                }
                black_box(&results);
            })
        });
    }
}

trait RunBaselineVincenty: Float + Sized {
    fn run(
        _g: &mut BenchmarkGroup<'_, WallTime>,
        _a_lats: &[Self],
        _a_lons: &[Self],
        _b_lats: &[Self],
        _b_lons: &[Self],
    ) {
    }
}

impl<T: Float> RunBaselineVincenty for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a_lats: &[T], a_lons: &[T], b_lats: &[T], b_lons: &[T]) {
        let mut results = vec![T::zero(); a_lats.len()];
        group.bench_function("baseline", |bench| {
            bench.iter(|| {
                black_box(baseline_vincenty_batch(a_lats, a_lons, b_lats, b_lons, &mut results));
                black_box(&results);
            })
        });
    }
}

trait RunNumKongVincenty: Vincenty + Float + Sized {
    fn run(
        _g: &mut BenchmarkGroup<'_, WallTime>,
        _a_lats: &[Self],
        _a_lons: &[Self],
        _b_lats: &[Self],
        _b_lons: &[Self],
    ) {
    }
}

impl<T: Vincenty + Float> RunNumKongVincenty for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a_lats: &[T], a_lons: &[T], b_lats: &[T], b_lons: &[T]) {
        let mut results = vec![T::zero(); a_lats.len()];
        group.bench_function("numkong", |bench| {
            bench.iter(|| {
                black_box(Vincenty::vincenty(a_lats, a_lons, b_lats, b_lons, &mut results));
                black_box(&results);
            })
        });
    }
}

trait RunGeoVincenty: Sized {
    fn run(
        _g: &mut BenchmarkGroup<'_, WallTime>,
        _a_lats: &[Self],
        _a_lons: &[Self],
        _b_lats: &[Self],
        _b_lons: &[Self],
    ) {
    }
}

impl<T: GeoScalar> RunGeoVincenty for T {
    fn run(group: &mut BenchmarkGroup<'_, WallTime>, a_lats: &[T], a_lons: &[T], b_lats: &[T], b_lons: &[T]) {
        let points_a = to_geo_points(a_lats, a_lons);
        let points_b = to_geo_points(b_lats, b_lons);
        let mut results = vec![0.0f64; points_a.len()];
        group.bench_function("geo", |bench| {
            bench.iter(|| {
                for (i, (a, b)) in points_a.iter().zip(&points_b).enumerate() {
                    results[i] = Geodesic.distance(*a, *b);
                }
                black_box(&results);
            })
        });
    }
}

// endregion

// region: Generic helpers

fn bench_haversine_dtype<T>(c: &mut Criterion, rng: &mut impl Rng, dtype: &str, count: usize)
where
    T: GeoScalar + RunBaselineHaversine + RunNumKongHaversine + RunGeoHaversine,
{
    let name = format!("geospatial/haversine/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((count * std::mem::size_of::<T>() * 4) as u64));

    let a_lats = generate_random_coords(rng, count, T::from(-90.0).unwrap(), T::from(90.0).unwrap());
    let a_lons = generate_random_coords(rng, count, T::from(-180.0).unwrap(), T::from(180.0).unwrap());
    let b_lats = generate_random_coords(rng, count, T::from(-90.0).unwrap(), T::from(90.0).unwrap());
    let b_lons = generate_random_coords(rng, count, T::from(-180.0).unwrap(), T::from(180.0).unwrap());

    <T as RunNumKongHaversine>::run(&mut group, &a_lats, &a_lons, &b_lats, &b_lons);
    <T as RunBaselineHaversine>::run(&mut group, &a_lats, &a_lons, &b_lats, &b_lons);
    <T as RunGeoHaversine>::run(&mut group, &a_lats, &a_lons, &b_lats, &b_lons);
    group.finish();
}

fn bench_vincenty_dtype<T>(c: &mut Criterion, rng: &mut impl Rng, dtype: &str, count: usize)
where
    T: GeoScalar + RunBaselineVincenty + RunNumKongVincenty + RunGeoVincenty,
{
    let name = format!("geospatial/vincenty/{dtype}");
    if !should_run_benchmark(&name) {
        return;
    }

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes((count * std::mem::size_of::<T>() * 4) as u64));

    let a_lats = generate_random_coords(rng, count, T::from(-90.0).unwrap(), T::from(90.0).unwrap());
    let a_lons = generate_random_coords(rng, count, T::from(-180.0).unwrap(), T::from(180.0).unwrap());
    let b_lats = generate_random_coords(rng, count, T::from(-90.0).unwrap(), T::from(90.0).unwrap());
    let b_lons = generate_random_coords(rng, count, T::from(-180.0).unwrap(), T::from(180.0).unwrap());

    <T as RunNumKongVincenty>::run(&mut group, &a_lats, &a_lons, &b_lats, &b_lons);
    <T as RunBaselineVincenty>::run(&mut group, &a_lats, &a_lons, &b_lats, &b_lons);
    <T as RunGeoVincenty>::run(&mut group, &a_lats, &a_lons, &b_lats, &b_lons);
    group.finish();
}

// endregion

// region: Benchmarks

/// Benchmark Haversine distance
pub fn bench_haversine(c: &mut Criterion) {
    let count = get_vector_dims();
    let mut rng = rand::rng();
    bench_haversine_dtype::<f32>(c, &mut rng, "f32", count);
    bench_haversine_dtype::<f64>(c, &mut rng, "f64", count);
}

/// Benchmark Vincenty distance
pub fn bench_vincenty(c: &mut Criterion) {
    let count = get_vector_dims();
    let mut rng = rand::rng();
    bench_vincenty_dtype::<f32>(c, &mut rng, "f32", count);
    bench_vincenty_dtype::<f64>(c, &mut rng, "f64", count);
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_baseline_correctness() {
        // New York to London: ~5570 km
        let distance = baseline_haversine(40.7128f64, -74.0060, 51.5074, -0.1278);
        assert!((distance - 5570.0).abs() < 50.0);
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
            (distance - 5570.0).abs() < 50.0,
            "Expected ~5570 km, got {} km",
            distance
        );
    }

    #[test]
    fn vincenty_baseline_same_point() {
        let distance = baseline_vincenty(0.0f64, 0.0, 0.0, 0.0);
        assert!(distance.abs() < 1e-6);
    }
}

// endregion

// region: Main

criterion_group! {
    name = benches;
    config = utils::configure_criterion();
    targets = bench_haversine, bench_vincenty
}
criterion_main!(benches);

// endregion
