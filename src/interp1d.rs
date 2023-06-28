use std::ops::{Index, Sub};

use ndarray::{ArrayBase, Data, Ix1, RawData};
use num_traits::{Float, Num, NumCast, PrimInt, ToPrimitive};

#[derive(Debug)]
pub enum InterpolationStrategy {
    Linear,
}

#[derive(Debug)]
pub struct Interp1D<S, T>
where
    S: RawData<Elem = T> + Data,
{
    /// x values are guaranteed to be strict monotonically rising
    /// if x is None, the x values are assumed to be the index of y
    x: Option<ArrayBase<S, Ix1>>,
    y: ArrayBase<S, Ix1>,
    strategy: InterpolationStrategy,
}

impl<S, T> Interp1D<S, T>
where
    S: RawData<Elem = T> + Data,
    T: Float,
{
    pub fn interp(&self, x: &T) -> T {
        match self.strategy {
            InterpolationStrategy::Linear => self.linear(x),
        }
    }

    fn linear(&self, x: &T) -> T {
        let idx = self.get_left_index(x);
        let (x1, y1) = self.get_point(idx);
        let (x2, y2) = self.get_point(idx + 1);
        let m = (y2 - y1) / (x2 - x1);
        let b = y1 - m * x1;
        m * *x + b
    }

    /// get x,y coordinate at given index
    /// panics at index out of range
    fn get_point(&self, idx: usize) -> (T, T) {
        match &self.x {
            Some(x) => (
                *x.get(idx).unwrap_or_else(|| unreachable!()),
                *self.y.get(idx).unwrap_or_else(|| unreachable!()),
            ),
            None => (
                NumCast::from(idx).unwrap_or_else(|| unreachable!()),
                *self.y.get(idx).unwrap_or_else(|| unreachable!()),
            ),
        }
    }

    /// the index of known value left of, or at x
    fn get_left_index(&self, x: &T) -> usize {
        if let Some(xs) = &self.x {
            // do bisection
            let mut range = (0usize, xs.len());
            while range.0 < range.1 + 1 {
                let mid_idx = (range.1 - range.0) / 2 + range.0;
                let mid_x = xs.get(mid_idx).unwrap_or_else(|| unreachable!());
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
        } else {
            x.floor().to_usize().unwrap()
        }
    }
}
