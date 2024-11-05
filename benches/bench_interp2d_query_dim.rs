use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array1, Axis};
use ndarray_interp::interp2d::{Bilinear, Interp2D, Interp2DScalar};

use rand_extensions::RandArray;

mod rand_extensions;

fn setup() -> (Interp2DScalar<f64, Bilinear>, Array1<f64>, Array1<f64>) {
    let data = Array::from_rand(10_000, (0.0, 1.0), 42)
        .into_shape_with_order((100, 100))
        .unwrap();
    let interp = Interp2D::builder(data).build().unwrap();
    let query_x = Array::from_rand(10_000, (0.0, 99.0), 123);
    let query_y = Array::from_rand(10_000, (0.0, 99.0), 96);
    (interp, query_x, query_y)
}

fn bench_scalar_data_1d_query(c: &mut Criterion) {
    let (interp, query_x, query_y) = black_box(setup());
    let query_x = query_x.into_shape_with_order((2500, 4)).unwrap();
    let query_y = query_y.into_shape_with_order((2500, 4)).unwrap();
    let query_arrx: Vec<_> = query_x.axis_iter(Axis(0)).collect();
    let query_arry: Vec<_> = query_y.axis_iter(Axis(0)).collect();

    c.bench_function("2D scalar `interp_array`", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array(x, y).unwrap();
            });
        })
    });

    let mut buffer = Array::zeros(4);
    c.bench_function("2D scalar `interp_array_into`", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array_into(x, y, buffer.view_mut()).unwrap();
            });
        })
    });
}

fn bench_scalar_data_2d_query(c: &mut Criterion) {
    let (interp, query_x, query_y) = black_box(setup());
    let query_x = query_x.into_shape_with_order((625, 4, 4)).unwrap();
    let query_y = query_y.into_shape_with_order((625, 4, 4)).unwrap();
    let query_arrx: Vec<_> = query_x.axis_iter(Axis(0)).collect();
    let query_arry: Vec<_> = query_y.axis_iter(Axis(0)).collect();

    c.bench_function("2D scalar `interp_array` 2D-query", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array(x, y).unwrap();
            });
        })
    });

    let mut buffer = Array::zeros((4, 4));
    c.bench_function("2D scalar `interp_array_into` 2D-query", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array_into(x, y, buffer.view_mut()).unwrap();
            });
        })
    });
}

fn bench_scalar_data_3d_query(c: &mut Criterion) {
    let (interp, query_x, query_y) = black_box(setup());
    let query_x = query_x.into_shape_with_order((125, 5, 4, 4)).unwrap();
    let query_y = query_y.into_shape_with_order((125, 5, 4, 4)).unwrap();
    let query_arrx: Vec<_> = query_x.axis_iter(Axis(0)).collect();
    let query_arry: Vec<_> = query_y.axis_iter(Axis(0)).collect();

    c.bench_function("2D scalar `interp_array` 3D-query", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array(x, y).unwrap();
            });
        })
    });

    let mut buffer = Array::zeros((5, 4, 4));
    c.bench_function("2D scalar `interp_array_into` 3D-query", |b| {
        b.iter(|| {
            query_arrx.iter().zip(query_arry.iter()).for_each(|(x, y)| {
                interp.interp_array_into(x, y, buffer.view_mut()).unwrap();
            });
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
