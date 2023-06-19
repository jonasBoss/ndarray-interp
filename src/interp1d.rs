use std::ops::Sub;

use ndarray::{RawData, Data, Ix1, ArrayBase};


pub enum InterpolationStrategy {
    Linear,
}

pub struct Interp1D<S, T>
where
    S: RawData<Elem = T> + Data,
{
    x: Option<ArrayBase<S, Ix1>>,
    y: ArrayBase<S, Ix1>,
    strategy: InterpolationStrategy,
}

impl<S, T> Interp1D<S, T>
where
    S: RawData<Elem = T> + Data,
    T: Into<usize>,
{
    fn get_left_index(&self, x) -> usize{
        if let Some(x) = &self.x {
            todo!()
        } else {
            x.floor() as usize
        }
    }
    pub fn interp(&self, x: T)->T{
        todo!()
    }
}