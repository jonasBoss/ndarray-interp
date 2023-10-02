//! Benchmarks for the [`Interp1D::interp_array`] method with different qeury dimensions

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, Axis};
use ndarray_interp::interp1d::Interp1D;

use rand_extensions::RandArray;

mod rand_extensions;

fn bench_scalar_data_1d_query(c: &mut Criterion) {
    let data = Array::from_rand(100, (0.0, 1.0), 42);
    let interp = Interp1D::builder(data).build().unwrap();
    let query = Array::from_rand(10_000, (0.0, 99.0), 123);

    let query = query.into_shape((2500, 4)).unwrap();
    let query_arr: Vec<_> = query.axis_iter(Axis(0)).collect();
    c.bench_function("1D scalar `interp_array`", |b| {
        b.iter(|| {
            for x in &query_arr {
                interp.interp_array(x).unwrap();
            }
        })
    });
}

fn bench_scalar_data_2d_query(c: &mut Criterion) {
    let data = Array::from_rand(100, (0.0, 1.0), 42);
    let interp = Interp1D::builder(data).build().unwrap();
    let query = Array::from_rand(10_000, (0.0, 99.0), 123);

    let query = query.into_shape((625, 4, 4)).unwrap();
    let query_arr: Vec<_> = query.axis_iter(Axis(0)).collect();
    c.bench_function("1D scalar `interp_array` 2D-query", |b| {
        b.iter(|| {
            for x in &query_arr {
                interp.interp_array(x).unwrap();
            }
        })
    });
}

fn bench_scalar_data_3d_query(c: &mut Criterion) {
    let data = Array::from_rand(100, (0.0, 1.0), 42);
    let interp = Interp1D::builder(data).build().unwrap();
    let query = Array::from_rand(10_000, (0.0, 99.0), 123);

    let query = query.into_shape((125, 5, 4, 4)).unwrap();
    let query_arr: Vec<_> = query.axis_iter(Axis(0)).collect();
    c.bench_function("1D scalar `interp_array` 3D-query", |b| {
        b.iter(|| {
            for x in &query_arr {
                interp.interp_array(x).unwrap();
            }
        })
    });
}

criterion_group!(
    benches,
    bench_scalar_data_1d_query,
    bench_scalar_data_2d_query,
    bench_scalar_data_3d_query,
);
criterion_main!(benches);
