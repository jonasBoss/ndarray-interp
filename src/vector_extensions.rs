use std::ops::Sub;

use ndarray::{Array1, ArrayView1};

///! This module contains the vector extensions trait

pub trait VectorExtensions {
    /// get the monotonic property of the vector
    fn monotonic_prop(&self) -> Monotonic;

    /// are the values liearly spaced
    fn is_linspaced(&self) -> bool;
}

/// Describes the monotonic property of a vector
pub enum Monotonic {
    Rising { strict: bool },
    Falling { strict: bool },
    NotMonotonic,
}

impl<T: PartialOrd + Sub<Output = T>> VectorExtensions for Array1<T> {
    fn monotonic_prop(&self) -> Monotonic {
        todo!()
    }

    fn is_linspaced(&self) -> bool {
        todo!()
    }
}

impl<T: PartialOrd + Sub<Output = T>> VectorExtensions for ArrayView1<'_, T> {
    fn monotonic_prop(&self) -> Monotonic {
        todo!()
    }

    fn is_linspaced(&self) -> bool {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use ndarray::{array, s, Array1};

    use super::{Monotonic, VectorExtensions};

    macro_rules! test_monotonic {
        ($d:ident, $expected:pat) => {
            match $d.monotonic_prop() {
                $expected => (),
                _ => panic!(),
            };
            match $d.slice(s![..;1]).monotonic_prop() {
                $expected => (),
                _ => panic!(),
            };
        };
    }

    // test with f64
    #[test]
    fn test_strict_monotonic_rising_f64() {
        let data: Array1<f64> = array![1.1, 2.0, 3.123, 4.5];
        test_monotonic!(data, Monotonic::Rising { strict: true });
    }

    #[test]
    fn test_monotonic_rising_f64() {
        let data: Array1<f64> = array![1.1, 2.0, 3.123, 3.123, 4.5];
        test_monotonic!(data, Monotonic::Rising { strict: false });
    }

    #[test]
    fn test_strict_monotonic_falling_f64() {
        let data: Array1<f64> = array![5.8, 4.123, 3.1, 2.0, 1.0];
        test_monotonic!(data, Monotonic::Falling { strict: true });
    }

    #[test]
    fn test_monotonic_falling_f64() {
        let data: Array1<f64> = array![5.8, 4.123, 3.1, 3.1, 2.0, 1.0];
        test_monotonic!(data, Monotonic::Falling { strict: false });
    }

    #[test]
    fn test_not_monotonic_f64() {
        let data: Array1<f64> = array![1.1, 2.0, 3.123, 3.120, 4.5];
        test_monotonic!(data, Monotonic::NotMonotonic);
    }

    // test with i32
    #[test]
    fn test_strict_monotonic_rising_i32() {
        let data: Array1<i32> = array![1, 2, 3, 4, 5];
        test_monotonic!(data, Monotonic::Rising { strict: true });
    }

    #[test]
    fn test_monotonic_rising_i32() {
        let data: Array1<i32> = array![1, 2, 3, 3, 4, 5];
        test_monotonic!(data, Monotonic::Rising { strict: false });
    }

    #[test]
    fn test_strict_monotonic_falling_i32() {
        let data: Array1<i32> = array![5, 4, 3, 2, 1];
        test_monotonic!(data, Monotonic::Falling { strict: true });
    }

    #[test]
    fn test_monotonic_falling_i32() {
        let data: Array1<i32> = array![5, 4, 3, 3, 2, 1];
        test_monotonic!(data, Monotonic::Falling { strict: false });
    }

    #[test]
    fn test_not_monotonic_i32() {
        let data: Array1<i32> = array![1, 2, 3, 2, 4, 5];
        test_monotonic!(data, Monotonic::NotMonotonic);
    }
}
