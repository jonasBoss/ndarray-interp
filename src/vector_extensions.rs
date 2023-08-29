use ndarray::{ArrayBase, Data, Ix1};
use std::fmt::Debug;

/// Helper methods for one dimensional numeric arrays
pub trait VectorExtensions<T> {
    /// get the monotonic property of the vector
    fn monotonic_prop(&self) -> Monotonic;

    /// Get the index of the next lower value inside the vector.
    /// This is not guaranteed to return the index of an exact match.
    ///
    /// This will never return the last index of the vector.
    /// when x is out of bounds it will either return `0` or `self.len() - 2`
    /// depending on which side it is out of bounds
    ///
    /// # Warning
    /// this method requires the [`monotonic_prop`](VectorExtensions::monotonic_prop) to be
    /// `Monotonic::Rising { strict: true }`
    /// otherwise the behaviour is undefined
    fn get_lower_index(&self, x: T) -> usize;
}

/// Describes the monotonic property of a vector
#[derive(Debug)]
pub enum Monotonic {
    Rising { strict: bool },
    Falling { strict: bool },
    NotMonotonic,
}
use num_traits::{cast, Num, NumCast};
use Monotonic::*;

use crate::interp1d::Linear;

impl<S> VectorExtensions<S::Elem> for ArrayBase<S, Ix1>
where
    S: Data,
    S::Elem: Debug + PartialOrd + Num + NumCast + Copy,
{
    fn monotonic_prop(&self) -> Monotonic {
        if self.len() <= 1 {
            return NotMonotonic;
        };

        self.windows(2)
            .into_iter()
            .try_fold(MonotonicState::start(), |state, items| {
                let a = items[0];
                let b = items[1];
                state.update(a, b).short_circuit()
            })
            .map_or_else(|mon| mon, |state| state.finish())
    }

    fn get_lower_index(&self, x: S::Elem) -> usize {
        // the vector should be strictly monotonic rising, otherwise we will
        // produce grabage

        // check in range, otherwise return the first or second last index
        // this allows for extrapolation
        if x <= self[0] {
            return 0;
        }
        if x >= self[self.len() - 1] {
            return self.len() - 2;
        }

        // We assume that the spacing is even. So we can calculate the index
        // and check it. This finishes in O(1) for even spaced axis.
        // Otherwise we do a binary search with O(log n)
        let mut range = (0usize, self.len() - 1);
        while range.0 + 1 < range.1 {
            let p1 = (
                self[range.0],
                cast(range.0)
                    .unwrap_or_else(|| unimplemented!("casting from usize should always work!")),
            );
            let p2 = (
                self[range.1],
                cast(range.1)
                    .unwrap_or_else(|| unimplemented!("casting from usize should always work!")),
            );

            let mid = Linear::calc_frac(p1, p2, x);
            let mid_idx: usize =
                cast(mid).unwrap_or_else(|| unimplemented!("failed to convert {mid:?} to usize"));

            let mid_x = self[mid_idx];

            if mid_x <= x && x < self[mid_idx + 1] {
                return mid_idx;
            }
            if mid_x < x {
                range.0 = mid_idx;
            } else {
                range.1 = mid_idx;
            }

            // the above alone has the potential to end in an infinte loop
            // do a binary search step to guarantee progress
            let mid_idx = (range.1 - range.0) / 2 + range.0;
            let mid_x = self[mid_idx];
            if mid_x == x {
                return mid_idx;
            }
            if mid_x < x {
                range.0 = mid_idx;
            } else {
                range.1 = mid_idx;
            }
        }
        range.0
    }
}

/// A State Machine for finding the monotonic property of an array
#[derive(Debug)]
enum MonotonicState {
    Init,
    NotStrict,
    Likely(Monotonic),
}

impl MonotonicState {
    /// start the state machine
    fn start() -> Self {
        Self::Init
    }

    /// update the state machine with two consecutive values from the vector
    fn update<T>(self, a: T, b: T) -> Self
    where
        T: PartialOrd,
    {
        use MonotonicState::*;
        match self {
            Init => {
                if a < b {
                    Likely(Rising { strict: true })
                } else if a == b {
                    NotStrict
                } else {
                    Likely(Falling { strict: true })
                }
            }
            NotStrict => {
                if a < b {
                    Likely(Rising { strict: false })
                } else if a == b {
                    NotStrict
                } else {
                    Likely(Falling { strict: false })
                }
            }
            Likely(Rising { strict }) => {
                if a == b {
                    Likely(Rising { strict: false })
                } else if a < b {
                    Likely(Rising { strict })
                } else {
                    Likely(NotMonotonic)
                }
            }
            Likely(Falling { strict }) => {
                if a == b {
                    Likely(Falling { strict: false })
                } else if a > b {
                    Likely(Falling { strict })
                } else {
                    Likely(NotMonotonic)
                }
            }
            Likely(NotMonotonic) => Likely(NotMonotonic),
        }
    }

    /// return `Err(Monotonic)` when the state machine can no longer change its state
    /// otherwise retruns `Ok(self)`
    ///
    /// This can be used with the [`Iterator::try_fold()`] method short-circuiting
    /// the iterator.
    fn short_circuit(self) -> Result<Self, Monotonic> {
        match self {
            MonotonicState::Likely(NotMonotonic) => Err(NotMonotonic),
            _ => Ok(self),
        }
    }

    /// get the final value of the state machine
    ///
    /// # panics
    /// when the state machine is still in the `Init` State
    fn finish(self) -> Monotonic {
        match self {
            MonotonicState::Init => panic!("`MonotonicState::update` was never called"),
            MonotonicState::NotStrict => NotMonotonic,
            MonotonicState::Likely(mon) => mon,
        }
    }
}

#[cfg(test)]
mod test {
    use ndarray::{array, s, Array, Array1};

    use super::{Monotonic, VectorExtensions};

    macro_rules! test_index {
        ($i:expr, $q:expr) => {
            let data = Array::linspace(0.0, 10.0, 11);
            assert_eq!($i, data.get_lower_index($q));
        };
        ($i:expr, $q:expr, exp) => {
            let data = Array::from_iter((0..11).map(|x| 2f64.powi(x)));
            assert_eq!($i, data.get_lower_index($q));
        };
    }

    #[test]
    fn test_outside_left() {
        test_index!(0, -1.0);
    }

    #[test]
    fn test_outside_right() {
        test_index!(9, 25.0);
    }

    #[test]
    fn test_left_border() {
        test_index!(0, 0.0);
    }

    #[test]
    fn test_right_border() {
        test_index!(9, 10.0);
    }

    #[test]
    fn test_exact_index() {
        for i in 0..10 {
            test_index!(i, i as f64);
        }
    }

    #[test]
    fn test_pos_inf_index() {
        test_index!(9, f64::INFINITY);
    }

    #[test]
    fn test_neg_inf_index() {
        test_index!(0, f64::NEG_INFINITY);
    }

    #[test]
    #[should_panic(expected = "not implemented: failed to convert NaN to usize")]
    fn test_nan() {
        test_index!(0, f64::NAN);
    }

    #[test]
    fn test_exponential_exact() {
        for (i, q) in (0..10).map(|x| (x as usize, 2f64.powi(x))) {
            test_index!(i, q, exp);
        }
    }

    #[test]
    fn test_exponential_right_border() {
        test_index!(9, 1024.0, exp);
    }

    macro_rules! test_monotonic {
        ($d:ident, $expected:pat) => {
            match $d.monotonic_prop() {
                $expected => (),
                value => panic!("{}", format!("got {value:?}")),
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

    #[test]
    fn test_ordered_view_on_unordred_array() {
        let data: Array1<i32> = array![5, 4, 3, 2, 1];
        let ordered = data.slice(s![..;-1]);
        test_monotonic!(ordered, Monotonic::Rising { strict: true });
    }

    #[test]
    fn test_starting_flat() {
        let data: Array1<i32> = array![1, 1, 2, 3, 4, 5];
        test_monotonic!(data, Monotonic::Rising { strict: false });
    }

    #[test]
    fn test_flat() {
        let data: Array1<i32> = array![1, 1, 1];
        test_monotonic!(data, Monotonic::NotMonotonic);
    }

    #[test]
    fn test_one_element_array() {
        let data: Array1<i32> = array![1];
        test_monotonic!(data, Monotonic::NotMonotonic);
    }
}
