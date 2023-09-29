use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
    Array, Axis,
};
use ndarray_interp::interp1d::Interp1D;

use rand_extensions::RandArray;

mod rand_extensions;

fn bench_interp1d_scalar(c: &mut Criterion) {
    let data = Array::from_rand(100, (0.0, 1.0), 42);
    let interp = Interp1D::builder(data).build().unwrap();
    let query = Array::from_rand(10_000, (0.0, 99.0), 123);

    c.bench_function("1D scalar `interp_scalar`", |b| {
        b.iter(|| {
            for &x in &query {
                interp.interp_scalar(x).unwrap();
            }
        })
    });

    c.bench_function("1D scalar `interp`", |b| {
        b.iter(|| {
            for &x in &query {
                interp.interp(x).unwrap();
            }
        })
    });

    c.bench_function("1D scalar `interp_array` 1D-long", |b| {
        b.iter(|| {
            interp.interp_array(&query).unwrap();
        })
    });

    let mut buffer = Array::zeros(1).remove_axis(Axis(0));
    c.bench_function("1D scalar `interp_into`", |b| {
        b.iter(|| {
            for &x in &query {
                interp.interp_into(x, buffer.view_mut()).unwrap();
            }
        })
    });
}

fn bench_interp1d_scalar_multithread(c: &mut Criterion) {
    let data = Array::from_rand(100, (0.0, 1.0), 42);
    let interp = Interp1D::builder(data).build().unwrap();
    let query = Array::from_rand(10_000, (0.0, 99.0), 123);

    c.bench_function("1D scalar MT `interp_scalar`", |b| {
        b.iter(|| {
            query.par_iter().for_each(|&x| {
                interp.interp_scalar(x).unwrap();
            });
        })
    });

    c.bench_function("1D scalar MT `interp`", |b| {
        b.iter(|| {
            query.par_iter().for_each(|&x| {
                interp.interp(x).unwrap();
            });
        })
    });

    let query = query.into_shape((2500, 4)).unwrap();
    let query_arr: Vec<_> = query.axis_iter(Axis(0)).collect();
    c.bench_function("1D scalar MT `interp_array`", |b| {
        b.iter(|| {
            query_arr.par_iter().for_each(|x| {
                interp.interp_array(x).unwrap();
            });
        })
    });
}

fn bench_interp1d_array(c: &mut Criterion) {
    let data = Array::from_rand(500, (0.0, 1.0), 69)
        .into_shape((100, 5))
        .unwrap();
    let interp = Interp1D::builder(data).build().unwrap();
    let query = Array::from_rand(10_000, (0.0, 99.0), 123);

    c.bench_function("1D array `interp`", |b| {
        b.iter(|| {
            for &x in &query {
                interp.interp(x).unwrap();
            }
        })
    });

    let mut buffer = Array::zeros(5);
    c.bench_function("1D array `interp_into`", |b| {
        b.iter(|| {
            for &x in &query {
                interp.interp_into(x, buffer.view_mut()).unwrap();
            }
        })
    });

    let query = query.into_shape((2500, 4)).unwrap();
    let query_arr: Vec<_> = query.axis_iter(Axis(0)).collect();
    c.bench_function("1D array `interp_array`", |b| {
        b.iter(|| {
            for x in &query_arr {
                interp.interp_array(x).unwrap();
            }
        })
    });

    let mut buffer = Array::zeros((4, 5));
    c.bench_function("1D array `interp_array_into`", |b| {
        b.iter(|| {
            for x in &query_arr {
                interp.interp_array_into(x, buffer.view_mut()).unwrap();
            }
        })
    });
}

criterion_group!(
    benches,
    bench_interp1d_scalar,
    bench_interp1d_array,
    bench_interp1d_scalar_multithread
);
criterion_main!(benches);
