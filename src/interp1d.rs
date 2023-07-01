use std::fmt::Debug;

use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, NdIndex};
use num_traits::{Num, NumCast};
use thiserror::Error;

use crate::vector_extensions::{Monotonic, VectorExtensions};

#[derive(Debug)]
pub enum InterpolationStrategy {
    Linear { extrapolate: bool },
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
pub struct Interp1D<Sd, Sx>
where
    Sd: Data,
    Sd::Elem: Num + Debug,
    Sx: Data<Elem = Sd::Elem>,
{
    /// x values are guaranteed to be strict monotonically rising
    /// if x is None, the x values are assumed to be the index of data
    x: Option<ArrayBase<Sx, Ix1>>,
    data: ArrayBase<Sd, Ix1>,
    strategy: InterpolationStrategy,
    range: (Sx::Elem, Sx::Elem),
}

impl<Sd, Sx> Interp1D<Sd, Sx>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug,
    Sx: Data<Elem = Sd::Elem>,
{
    /// Get the [[Interp1DBuilder]]
    pub fn builder(data: ArrayBase<Sd, Ix1>) -> Interp1DBuilder<Sd, Sx> {
        Interp1DBuilder::new(data)
    }

    /// Interpolated value at x
    pub fn interp(&self, x: Sx::Elem) -> Result<Sd::Elem, InterpolateError> {
        match &self.strategy {
            Linear { .. } => self.linear(x),
        }
    }

    /// Interpolate values at xs
    pub fn interp_array<D>(
        &self,
        xs: &ArrayBase<Sx, D>,
    ) -> Result<Array<Sd::Elem, D>, InterpolateError>
    where
        D: Dimension,
        D::Pattern: NdIndex<D>,
    {
        let ys = Array::zeros(xs.raw_dim());
        xs.indexed_iter().try_fold(ys, |mut ys, (idx, x)| {
            let y_ref = ys.get_mut(idx).unwrap_or_else(|| unreachable!());
            *y_ref = self.interp(*x)?;
            Ok(ys)
        })
    }

    /// the implementation for [Linear] strategy
    fn linear(&self, x: Sx::Elem) -> Result<Sd::Elem, InterpolateError> {
        if matches!(self.strategy, Linear { extrapolate: false })
            && !(self.range.0 <= x && x <= self.range.1)
        {
            return Err(InterpolateError::OutOfBounds(format!(
                "x = {x:#?} is not in range of {:#?}",
                self.range
            )));
        }
        let mut idx = self.get_left_index(x);
        if idx == self.data.len() - 1 {
            idx -= 1;
        }
        Ok(Self::calc_frac(
            self.get_point(idx),
            self.get_point(idx + 1),
            x,
        ))
    }

    /// get x,data coordinate at given index
    /// panics at index out of range
    fn get_point(&self, idx: usize) -> (Sx::Elem, Sd::Elem) {
        match &self.x {
            Some(x) => (
                *x.get(idx).unwrap_or_else(|| unreachable!()),
                *self.data.get(idx).unwrap_or_else(|| unreachable!()),
            ),
            None => (
                NumCast::from(idx).unwrap_or_else(|| unreachable!()),
                *self.data.get(idx).unwrap_or_else(|| unreachable!()),
            ),
        }
    }

    /// linearly interpolate/exrapolate between two points
    fn calc_frac(
        (x1, y1): (Sx::Elem, Sd::Elem),
        (x2, y2): (Sx::Elem, Sd::Elem),
        x: Sx::Elem,
    ) -> Sx::Elem {
        let b = y1;
        let m = (y2 - y1) / (x2 - x1);
        m * (x - x1) + b
    }

    /// the index of known value left of, or at x
    fn get_left_index(&self, x: Sx::Elem) -> usize {
        if let Some(xs) = &self.x {
            // the x axis is given so we need to search for the index, and can not calculate it.
            // the x axis is guaranteed to be strict monotonically rising.
            // We assume that the spacing is even. So we can calculate the index
            // and check it. This finishes in O(1) for even spaced axis.
            // Otherwise we do a binary search with O(log n)
            let mut range = (0usize, xs.len() - 1);
            while range.0 + 1 < range.1 {
                let p1 = (
                    *xs.get(range.0).unwrap_or_else(|| unreachable!()),
                    NumCast::from(range.0).unwrap_or_else(|| {
                        unimplemented!("casting from usize should always work!")
                    }),
                );
                let p2 = (
                    *xs.get(range.1).unwrap_or_else(|| unreachable!()),
                    NumCast::from(range.1).unwrap_or_else(|| {
                        unimplemented!("casting from usize should always work!")
                    }),
                );

                let mid = Self::calc_frac(p1, p2, x);
                if mid < NumCast::from(0).unwrap_or_else(|| unimplemented!()) {
                    // neagtive values might occure when extrapolating index 0 is
                    // the guaranteed solution
                    return 0;
                }

                let mut mid_idx: usize = NumCast::from(mid).unwrap_or_else(|| {
                    unimplemented!("mid is positive, so this should work always")
                });
                if mid_idx == range.1 {
                    mid_idx -= 1
                };
                let mut mid_x = xs.get(mid_idx).unwrap_or_else(|| unreachable!());

                if mid_x <= &x && x <= *xs.get(mid_idx + 1).unwrap_or_else(|| unreachable!()) {
                    return mid_idx;
                }
                if mid_x < &x {
                    range.0 = mid_idx;
                } else {
                    range.1 = mid_idx;
                }

                // the above alone has the potential to end in an infinte loop
                // do a binary search step to guarantee progress
                mid_idx = (range.1 - range.0) / 2 + range.0;
                mid_x = xs.get(mid_idx).unwrap_or_else(|| unreachable!());
                if mid_x == &x {
                    return mid_idx;
                }
                if mid_x < &x {
                    range.0 = mid_idx;
                } else {
                    range.1 = mid_idx;
                }
            }
            range.0
        } else if x < NumCast::from(0).unwrap_or_else(|| unimplemented!()) {
            0
        } else {
            // this relies on the fact that float -> int cast will return the next lower int
            // for positive values
            let x = NumCast::from(x)
                .unwrap_or_else(|| unimplemented!("x is positive, so this should always work"));
            if x > self.data.len() - 1 {
                self.data.len() - 1
            } else {
                x
            }
        }
    }
}

/// Create and configure a [Interp1D] Interpolator.
#[derive(Debug)]
pub struct Interp1DBuilder<Sd, Sx>
where
    Sd: Data,
    Sd::Elem: Num + Debug,
    Sx: Data<Elem = Sd::Elem>,
{
    x: Option<ArrayBase<Sx, Ix1>>,
    data: ArrayBase<Sd, Ix1>,
    strategy: InterpolationStrategy,
}

impl<Sd, Sx> Interp1DBuilder<Sd, Sx>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug,
    Sx: Data<Elem = Sd::Elem>,
{
    /// Create a new [Interp1DBuilder] and provide the data to interpolate.
    /// When nothing else is configured [Interp1DBuilder::build] will create an Interpolator using
    /// Linear Interpolation without extrapolation. As x axis the index to the data would be used.
    pub fn new(data: ArrayBase<Sd, Ix1>) -> Self {
        Interp1DBuilder {
            x: None,
            data,
            strategy: Linear { extrapolate: false },
        }
    }

    /// Add an custom x axis for the data. The axis needs to have the same lenght
    /// and store the same Type as the data.
    /// If the x axis is not set the index `0..y.len()` is used
    pub fn x(mut self, x: ArrayBase<Sx, Ix1>) -> Self {
        self.x = Some(x);
        self
    }

    /// Set the [InterpolationStrategy]. By default [Linear] with `Linear{extrapolate: false}` is used.
    pub fn strategy(mut self, strategy: InterpolationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Validate input data and create the configured [Interp1D]
    pub fn build(self) -> Result<Interp1D<Sd, Sx>, BuilderError> {
        match &self.strategy {
            Linear { .. } => {
                if self.data.len() < 2 {
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
            if self.data.len() != x.len() {
                Err(BuilderError::AxisLenght(format!(
                    "Lengths of x and data axis need to match. Got x: {:}, data: {:}",
                    x.len(),
                    self.data.len()
                )))
            } else {
                Ok(())
            }?;
        }
        let range = match &self.x {
            Some(x) => (
                *x.first().unwrap_or_else(|| unreachable!()),
                *x.last().unwrap_or_else(|| unreachable!()),
            ),
            None => (
                NumCast::from(0).unwrap_or_else(|| unimplemented!()),
                NumCast::from(self.data.len() - 1).unwrap_or_else(|| unimplemented!()),
            ),
        };
        Ok(Interp1D {
            x: self.x,
            data: self.data,
            strategy: self.strategy,
            range,
        })
    }
}

#[cfg(test)]
mod test {
    use ndarray::array;
    use ndarray::s;
    use ndarray::OwnedRepr;
    use num_traits::NumCast;

    use super::Interp1D;
    use super::Interp1DBuilder;
    use super::InterpolationStrategy::*;
    use crate::BuilderError;
    use crate::InterpolateError;

    #[test]
    fn test_type_cast_assumptions() {
        assert_eq!(<i32 as NumCast>::from(1.75).unwrap(), 1);
        assert_eq!(<i32 as NumCast>::from(1.25).unwrap(), 1);
    }

    #[test]
    fn interp_y_only() {
        let interp: Interp1D<_, OwnedRepr<_>> =
            Interp1D::builder(array![1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.5])
                .build()
                .unwrap();
        assert_eq!(interp.interp(0.0).unwrap(), 1.5);
        assert_eq!(interp.interp(9.0).unwrap(), 10.5);
        assert_eq!(interp.interp(4.5).unwrap(), 6.0);
        assert_eq!(interp.interp(0.25).unwrap(), 1.625);
        assert_eq!(interp.interp(8.75).unwrap(), 10.125);
    }

    #[test]
    fn extrapolate_y_only() {
        let interp: Interp1D<_, OwnedRepr<_>> = Interp1D::builder(array![1.0, 2.0, 1.5])
            .strategy(Linear { extrapolate: true })
            .build()
            .unwrap();
        assert_eq!(interp.interp(-1.0).unwrap(), 0.0);
        assert_eq!(interp.interp(3.0).unwrap(), 1.0);
    }

    #[test]
    fn interp_with_x_and_y() {
        let interp =
            Interp1DBuilder::new(array![1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.5])
                .x(array![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                .strategy(Linear { extrapolate: false })
                .build()
                .unwrap();
        assert_eq!(interp.interp(-4.0).unwrap(), 1.5);
        assert_eq!(interp.interp(5.0).unwrap(), 10.5);
        assert_eq!(interp.interp(0.5).unwrap(), 6.0);
        assert_eq!(interp.interp(-3.75).unwrap(), 1.625);
        assert_eq!(interp.interp(4.75).unwrap(), 10.125);
    }

    #[test]
    fn interp_with_x_and_y_expspaced() {
        let interp = Interp1DBuilder::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
            .x(array![
                1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0
            ])
            .strategy(Linear { extrapolate: false })
            .build()
            .unwrap();
        assert_eq!(interp.interp(1.0).unwrap(), 1.0);
        assert_eq!(interp.interp(512.0).unwrap(), 1.0);
        assert_eq!(interp.interp(42.0).unwrap(), 4.6875);
        assert_eq!(interp.interp(365.0).unwrap(), 1.57421875);
    }

    #[test]
    fn extrapolate_with_x_and_y() {
        let interp = Interp1DBuilder::new(array![1.0, 0.0, 1.5])
            .x(array![0.0, 1.0, 1.5])
            .strategy(Linear { extrapolate: true })
            .build()
            .unwrap();
        assert_eq!(interp.interp(-1.0).unwrap(), 2.0);
        assert_eq!(interp.interp(2.0).unwrap(), 3.0);
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

    #[test]
    fn interp_y_only_out_of_bounds() {
        let interp: Interp1D<_, OwnedRepr<_>> =
            Interp1D::builder(array![1.0, 2.0, 3.0]).build().unwrap();
        assert!(matches!(
            interp.interp(-0.1),
            Err(InterpolateError::OutOfBounds(_))
        ));
        assert!(matches!(
            interp.interp(9.0),
            Err(InterpolateError::OutOfBounds(_))
        ));
    }

    #[test]
    fn interp_with_x_and_y_out_of_bounds() {
        let interp = Interp1DBuilder::new(array![1.0, 2.0, 3.0])
            .x(array![-4.0, -3.0, 2.0])
            .strategy(Linear { extrapolate: false })
            .build()
            .unwrap();
        assert!(matches!(
            interp.interp(-4.1),
            Err(InterpolateError::OutOfBounds(_))
        ));
        assert!(matches!(
            interp.interp(2.1),
            Err(InterpolateError::OutOfBounds(_))
        ));
    }

    #[test]
    fn interp_builder_errors() {
        assert!(matches!(
            Interp1DBuilder::<_, OwnedRepr<_>>::new(array![1]).build(),
            Err(BuilderError::NotEnoughData(_))
        ));
        assert!(matches!(
            Interp1DBuilder::new(array![1, 2])
                .x(array![1, 2, 3])
                .build(),
            Err(BuilderError::AxisLenght(_))
        ));
        assert!(matches!(
            Interp1DBuilder::new(array![1, 2, 3])
                .x(array![1, 2, 2])
                .build(),
            Err(BuilderError::Monotonic(_))
        ));
    }

    #[test]
    fn interp_view_array() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let interp = Interp1D::builder(a.slice(s![..;-1]))
            .x(array![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .build()
            .unwrap();
        assert_eq!(interp.interp(-4.0).unwrap(), 10.0);
        assert_eq!(interp.interp(5.0).unwrap(), 1.0);
        assert_eq!(interp.interp(0.0).unwrap(), 6.0);
        assert_eq!(interp.interp(-3.5).unwrap(), 9.5);
        assert_eq!(interp.interp(4.75).unwrap(), 1.25);
    }
}