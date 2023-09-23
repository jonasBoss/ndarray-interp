use ndarray::{Array, Ix1};
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    rngs::StdRng,
    Rng, SeedableRng,
};

fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

pub trait RandArray<T> {
    fn from_rand(size: usize, range: (T, T), seed: u64) -> Self;
    fn from_rand_ordered(size: usize, range: (T, T), seed: u64) -> Self;
}

impl<T: SampleUniform + PartialOrd> RandArray<T> for Array<T, Ix1> {
    fn from_rand(size: usize, range: (T, T), seed: u64) -> Self {
        Array::from_iter(
            rng(seed)
                .sample_iter(Uniform::new_inclusive(range.0, range.1))
                .take(size),
        )
    }

    fn from_rand_ordered(size: usize, range: (T, T), seed: u64) -> Self {
        let mut arr = Vec::from_iter(
            rng(seed)
                .sample_iter(Uniform::new_inclusive(range.0, range.1))
                .take(size),
        );
        arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        arr.dedup();
        Array::from(arr)
    }
}
