use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{
    parallel::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    Array, Axis, Zip,
};
use ndarray_interp::interp2d::Interp2D;

use rand_extensions::RandArray;

mod rand_extensions;

fn bench_interp2d_scalar(c: &mut Criterion) {
    let data = Array::from_rand(10_000, (0.0, 1.0), 42)
        .into_shape_with_order((100, 100))
        .unwrap();
    let interp = Interp2D::builder(data).build().unwrap();
    let query_x = Array::from_rand(10_000, (0.0, 99.0), 123);
    let query_y = Array::from_rand(10_000, (0.0, 99.0), 96);

    c.bench_function("2D scalar `interp_scalar`", |b| {
        b.iter(|| {
            query_x.iter().zip(query_y.iter()).for_each(|(&x, &y)| {
                interp.interp_scalar(x, y).unwrap();
            });
        })
    });

    c.bench_function("2D scalar `interp`", |b| {
        b.iter(|| {
            query_x.iter().zip(query_y.iter()).for_each(|(&x, &y)| {
                interp.interp(x, y).unwrap();
            });
        })
    });

    let mut buffer = Array::zeros(1).remove_axis(Axis(0));
    c.bench_function("2D scalar `interp_into`", |b| {
        b.iter(|| {
            query_x.iter().zip(query_y.iter()).for_each(|(&x, &y)| {
                interp.interp_into(x, y, buffer.view_mut()).unwrap();
            });
        })
    });
}

fn bench_interp2d_scalar_multithread(c: &mut Criterion) {
    let data = Array::from_rand(10_000, (0.0, 1.0), 42)
        .into_shape_with_order((100, 100))
        .unwrap();
    let interp = Interp2D::builder(data).build().unwrap();
    let query_x = Array::from_rand(10_000, (0.0, 99.0), 123);
    let query_y = Array::from_rand(10_000, (0.0, 99.0), 96);

    c.bench_function("2D scalar MT `interp_scalar`", |b| {
        b.iter(|| {
            Zip::from(&query_x).and(&query_y).par_for_each(|&x, &y| {
                interp.interp_scalar(x, y).unwrap();
            });
        })
    });

    c.bench_function("2D scalar MT `interp`", |b| {
        b.iter(|| {
            Zip::from(&query_x).and(&query_y).par_for_each(|&x, &y| {
                interp.interp(x, y).unwrap();
            });
        })
    });

    let query_x = query_x.into_shape_with_order((2500, 4)).unwrap();
    let query_y = query_y.into_shape_with_order((2500, 4)).unwrap();
    let query_arrx: Vec<_> = query_x.axis_iter(Axis(0)).collect();
    let query_arry: Vec<_> = query_y.axis_iter(Axis(0)).collect();
    c.bench_function("2D scalar MT `interp_array`", |b| {
        b.iter(|| {
            query_arrx
                .par_iter()
                .zip(query_arry.par_iter())
                .for_each(|(x, y)| {
                    interp.interp_array(x, y).unwrap();
                });
        })
    });
}

fn bench_interp2d_array(c: &mut Criterion) {
    let data = Array::from_rand(50_000, (0.0, 1.0), 69)
        .into_shape_with_order((100, 100, 5))
        .unwrap();
    let interp = Interp2D::builder(data).build().unwrap();
    let query_x = Array::from_rand(10_000, (0.0, 99.0), 123);
    let query_y = Array::from_rand(10_000, (0.0, 99.0), 96);

    c.bench_function("2D array `interp`", |b| {
        b.iter(|| {
            query_x.iter().zip(query_y.iter()).for_each(|(&x, &y)| {
                interp.interp(x, y).unwrap();
            });
        })
    });

    let mut buffer = Array::zeros(5);
    c.bench_function("2D array `interp_into`", |b| {
        b.iter(|| {
            query_x.iter().zip(query_y.iter()).for_each(|(&x, &y)| {
                interp.interp_into(x, y, buffer.view_mut()).unwrap();
            });
        })
    });

    let query_x = query_x.into_shape_with_order((2500, 4)).unwrap();
    let query_y = query_y.into_shape_with_order((2500, 4)).unwrap();
    let query_arrx: Vec<_> = query_x.axis_iter(Axis(0)).collect();
    let query_arry: Vec<_> = query_y.axis_iter(Axis(0)).collect();
    c.bench_function("2D array `interp_array`", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array(x, y).unwrap();
            });
        })
    });

    let mut buffer = Array::zeros((4, 5));
    c.bench_function("2D array `interp_array_into`", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array_into(x, y, buffer.view_mut()).unwrap();
            });
        })
    });
}

criterion_group!(
    benches,
    bench_interp2d_scalar,
    bench_interp2d_array,
    bench_interp2d_scalar_multithread
);
criterion_main!(benches);
