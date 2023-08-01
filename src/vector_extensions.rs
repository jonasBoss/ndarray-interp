use ndarray::{ArrayBase, Data, Ix1};

/// Helper methods for one dimensional arrays
pub(crate) trait VectorExtensions<T> {
    /// get the monotonic property of the vector
    fn monotonic_prop(&self) -> Monotonic;

    /// Get the index of the next lower value inside the vector
    ///
    /// This will never return the last index of the vector.
    /// when x is out of bounds it will either return index 0 or self.len() - 2
    ///
    /// # Warning
    /// this method requires the [`monotonic_prop`] to be
    /// `Monotonic::Rising { strict: true }`
    /// otherwise the behaviour is undefined
    fn get_lower_index(&self, x: T) -> usize;
}

/// Describes the monotonic property of a vector
#[derive(Debug)]
pub(crate) enum Monotonic {
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
    S::Elem: PartialOrd + Num + NumCast + Copy,
{
    fn monotonic_prop(&self) -> Monotonic {
        if self.len() <= 1 {
            return NotMonotonic;
        };

        #[derive(Debug)]
        enum State {
            Init,
            NotStrict,
            Known(Monotonic),
        }
        use State::*;

        let state = self
            .windows(2)
            .into_iter()
            .try_fold(Init, |state, items| {
                let a = items.get(0).unwrap_or_else(|| unreachable!());
                let b = items.get(1).unwrap_or_else(|| unreachable!());
                match state {
                    Init => {
                        if a < b {
                            return Ok(Known(Rising { strict: true }));
                        } else if a == b {
                            return Ok(NotStrict);
                        }
                        Ok(Known(Falling { strict: true }))
                    }
                    NotStrict => {
                        if a < b {
                            return Ok(Known(Rising { strict: false }));
                        } else if a == b {
                            return Ok(NotStrict);
                        }
                        Ok(Known(Falling { strict: false }))
                    }
                    Known(Rising { strict }) => {
                        if a == b {
                            return Ok(Known(Rising { strict: false }));
                        } else if a < b {
                            return Ok(Known(Rising { strict }));
                        }
                        Err(NotMonotonic)
                    }
                    Known(Falling { strict }) => {
                        if a == b {
                            return Ok(Known(Falling { strict: false }));
                        } else if a > b {
                            return Ok(Known(Falling { strict }));
                        }
                        Err(NotMonotonic)
                    }
                    Known(NotMonotonic) => unreachable!(),
                }
            })
            .unwrap_or(Known(NotMonotonic));

        if let Known(state) = state {
            state
        } else {
            NotMonotonic
        }
    }

    fn get_lower_index(&self, x: S::Elem) -> usize {
        // the vector should be strictly monotonic rising, otherwise we will
        // produce grabage
        //
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
            if mid < cast(0).unwrap_or_else(|| unimplemented!()) {
                // neagtive values might occure when extrapolating. index 0 is
                // the guaranteed solution
                return 0;
            }

            let mut mid_idx: usize = cast(mid)
                .unwrap_or_else(|| unimplemented!("mid is positive, so this should work always"));
            if mid_idx == range.1 {
                mid_idx -= 1;
            };
            let mut mid_x = self[mid_idx];

            if mid_x <= x && x <= self[mid_idx + 1] {
                return mid_idx;
            }
            if mid_x < x {
                range.0 = mid_idx;
            } else {
                range.1 = mid_idx;
            }

            // the above alone has the potential to end in an infinte loop
            // do a binary search step to guarantee progress
            mid_idx = (range.1 - range.0) / 2 + range.0;
            mid_x = self[mid_idx];
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

#[cfg(test)]
mod test {
    use ndarray::{array, s, Array1};

    use super::{Monotonic, VectorExtensions};

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
