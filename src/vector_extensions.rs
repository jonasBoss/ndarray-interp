use std::ops::Sub;

use ndarray::Array1;

///! This module contains the vector extensions trait

pub trait VectorExtensions {
    /// get the monotonic property of the vector
    fn monotonic_prop(&self) -> Monotonic;

    /// are the values liearly spaced
    fn is_linspaced(&self) -> bool;
}

/// Describes the monotonic property of a vector
pub enum Monotonic{
    Rising{strict: bool},
    Falling{strict: bool},
    NotMonotonic,
}

impl<T: PartialOrd + Sub<Output=T>> VectorExtensions for Array1<T>{
    fn monotonic_prop(&self) -> Monotonic {
        todo!()
    }

    fn is_linspaced(&self) -> bool {
        todo!()
    }
}