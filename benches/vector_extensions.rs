use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array1};
use ndarray_interp::vector_extensions::VectorExtensions;
use rand::{distributions::Uniform, prelude::*};

fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn run(arr: &Array1<f64>, query: &Array1<f64>) {
    for &x in query {
        arr.get_lower_index(x);
    }
}

fn uniform_rng() -> Array1<f64> {
    let mut arr = Vec::from_iter(
        rng(42)
            .sample_iter(Uniform::new_inclusive(0.0, 1.0))
            .take(100),
    );
    arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    arr.dedup();
    Array::from(arr)
}

fn bunched_linspace() -> Array1<f64> {
    let mut arr: Vec<f64> =
        Vec::from_iter(Array::linspace(0.0, 1.0, 20).into_iter().flat_map(|x| {
            rng(42)
                .sample_iter(Uniform::new_inclusive(-0.001, 0.001))
                .take(5)
                .map(move |noise| x + noise)
        }));
    arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    arr.dedup();
    Array::from(arr)
}

fn noisy_linspace() -> Array1<f64> {
    let mut arr = Array::linspace(0.0, 1.0, 100);
    arr.iter_mut()
        .zip(rng(42).sample_iter(Uniform::new(-0.002, 0.002)))
        .for_each(|(val, noise)| {
            *val += noise;
        });
    arr
}

fn bench_get_lower_index(c: &mut Criterion) {
    let query = Array::from_iter(
        rng(69)
            .sample_iter(Uniform::new_inclusive(-0.1, 1.1))
            .take(1000),
    );

    let arr = black_box(Array::linspace(0.0, 1.0, 100));
    c.bench_function("Linspaced", |b| {
        b.iter(|| run(&arr, &query));
    });

    let arr = uniform_rng();
    c.bench_function("Uniform rng", |b| {
        b.iter(|| run(&arr, &query));
    });

    let arr = bunched_linspace();
    c.bench_function("Linspace bunched", |b| {
        b.iter(|| run(&arr, &query));
    });

    let arr = noisy_linspace();
    c.bench_function("Linspace noisy", |b| {
        b.iter(|| run(&arr, &query));
    });

    let arr = Array::logspace(2.0, 0.0, 8.0, 100);
    let query = Array::from_iter(
        rng(69)
            .sample_iter(Uniform::new_inclusive(0.95, 256.5))
            .take(1000),
    );
    c.bench_function("Logspaced", |b| {
        b.iter(|| run(&arr, &query));
    });
}

criterion_group!(benches, bench_get_lower_index);
criterion_main!(benches);
