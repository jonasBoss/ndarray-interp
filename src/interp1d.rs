use std::fmt::Debug;

use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, NdIndex, RawData};
use num_traits::{Float, Num, NumCast};
use thiserror::Error;

use crate::vector_extensions::{Monotonic, VectorExtensions};

#[derive(Debug)]
pub enum InterpolationStrategy {
    Linear,
}
use InterpolationStrategy::*;

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("{0}")]
    NotEnoughData(String),
    #[error("{0}")]
    Monotonic(String),
    #[error("{0}")]
    AxisLenght(String),
}

#[derive(Debug, Error)]
pub enum InterpolateError {
    #[error("{0}")]
    OutOfBounds(String),
}

#[derive(Debug)]
pub struct Interp1DBuilder<S, T>
where
    S: RawData<Elem = T> + Data,
    T: Num,
{
    x: Option<ArrayBase<S, Ix1>>,
    y: ArrayBase<S, Ix1>,
    strategy: InterpolationStrategy,
}
impl<S, T> Interp1DBuilder<S, T>
where
    S: RawData<Elem = T> + Data,
    T: Num + PartialOrd + Clone + NumCast,
{
    pub fn new(y: ArrayBase<S, Ix1>) -> Self {
        Interp1DBuilder {
            x: None,
            y,
            strategy: Linear,
        }
    }
    pub fn x(mut self, x: ArrayBase<S, Ix1>) -> Self {
        self.x = Some(x);
        self
    }
    pub fn strategy(mut self, strategy: InterpolationStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    pub fn build(self) -> Result<Interp1D<S, T>, BuilderError> {
        match self.strategy {
            Linear => {
                if self.y.len() < 2 {
                    Err(BuilderError::NotEnoughData(
                        "Linear Interpolation needs at least two data points".into(),
                    ))
                } else {
                    Ok(())
                }
            }
        }?;

        if let Some(x) = &self.x {
            match x.monotonic_prop() {
                Monotonic::Rising { strict: true } => Ok(()),
                _ => Err(BuilderError::Monotonic(
                    "Values in the x axis need to be strictly monotonic rising".into(),
                )),
            }?;
            if self.y.len() != x.len() {
                Err(BuilderError::AxisLenght(format!(
                    "Lengths of x and y axis need to match. Got x: {:}, y: {:}",
                    x.len(),
                    self.y.len()
                )))
            } else {
                Ok(())
            }?;
        }
        let range = match &self.x {
            Some(x) => (
                x.first().unwrap_or_else(|| unreachable!()).clone(),
                x.last().unwrap_or_else(|| unreachable!()).clone(),
            ),
            None => (
                NumCast::from(0).unwrap_or_else(|| unimplemented!()),
                NumCast::from(self.y.len() - 1).unwrap_or_else(|| unimplemented!()),
            ),
        };
        Ok(Interp1D {
            x: self.x,
            y: self.y,
            strategy: self.strategy,
            range,
        })
    }
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
    range: (T, T),
}

impl<S, T> Interp1D<S, T>
where
    S: RawData<Elem = T> + Data,
    T: PartialOrd + Num + Clone + NumCast,
{
    pub fn builder(y: ArrayBase<S, Ix1>) -> Interp1DBuilder<S, T> {
        Interp1DBuilder::new(y)
    }
}

impl<S, T> Interp1D<S, T>
where
    S: RawData<Elem = T> + Data,
    T: Float + Debug,
{
    /// Interpolated value at x
    pub fn interp(&self, x: T) -> Result<T, InterpolateError> {
        match self.strategy {
            Linear => self.linear(x),
        }
    }

    /// Interpolate values at xs
    pub fn interp_array<D>(&self, xs: &ArrayBase<S, D>) -> Result<Array<T, D>, InterpolateError>
    where
        D: Dimension,
        <D as Dimension>::Pattern: NdIndex<D>,
    {
        let ys = Array::zeros(xs.raw_dim());
        xs.indexed_iter().try_fold(ys, |mut ys, (idx, x)| {
            let y_ref = ys.get_mut(idx).unwrap_or_else(|| unreachable!());
            *y_ref = self.interp(*x)?;
            Ok(ys)
        })
    }

    fn linear(&self, x: T) -> Result<T, InterpolateError> {
        if !(self.range.0 <= x && x <= self.range.1) {
            return Err(InterpolateError::OutOfBounds(format!(
                "x = {x:#?} is not in range of {:#?}",
                self.range
            )));
        }
        let idx = self.get_left_index(x);
        let (x1, y1) = self.get_point(idx);
        let (x2, y2) = self.get_point(idx + 1);
        let m = (y2 - y1) / (x2 - x1);
        let b = y1 - m * x1;
        Ok(m * x + b)
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
    fn get_left_index(&self, x: T) -> usize {
        if let Some(xs) = &self.x {
            // do bisection
            let mut range = (0usize, xs.len() - 1);
            while range.0 + 1 < range.1 {
                let mid_idx = (range.1 - range.0) / 2 + range.0;
                let mid_x = *xs.get(mid_idx).unwrap_or_else(|| unreachable!());
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
            match x.ceil().to_usize().unwrap() {
                0 => 0,     // avoid out of bounds left
                x => x - 1, // avoid out of bounds right
            }
        }
    }
}

#[cfg(test)]
mod test {
    use ndarray::array;

    use super::Interp1D;
    use super::Interp1DBuilder;
    use super::InterpolationStrategy::*;

    #[test]
    fn interp_y_only() {
        let interp = Interp1D::builder(array![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
            .build()
            .unwrap();
        assert_eq!(interp.interp(0.0).unwrap(), 1.0);
        assert_eq!(interp.interp(9.0).unwrap(), 1.0);
        assert_eq!(interp.interp(4.5).unwrap(), 5.0);
        assert_eq!(interp.interp(0.5).unwrap(), 1.5);
        assert_eq!(interp.interp(8.75).unwrap(), 1.25);
    }

    #[test]
    fn interp_with_x_and_y() {
        let interp = Interp1DBuilder::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
            .x(array![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .strategy(Linear)
            .build()
            .unwrap();
        assert_eq!(interp.interp(-4.0).unwrap(), 1.0);
        assert_eq!(interp.interp(5.0).unwrap(), 1.0);
        assert_eq!(interp.interp(0.0).unwrap(), 5.0);
        assert_eq!(interp.interp(-3.5).unwrap(), 1.5);
        assert_eq!(interp.interp(4.75).unwrap(), 1.25);
    }

    #[test]
    fn interp_array() {
        let interp = Interp1D::builder(array![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
            .build()
            .unwrap();
        let x_query = array![[1.0, 2.0, 9.0], [4.0, 5.0, 7.5]];
        let y_expect = array![[2.0, 3.0, 1.0], [5.0, 5.0, 2.5]];
        assert_eq!(interp.interp_array(&x_query).unwrap(), y_expect);
    }
}
