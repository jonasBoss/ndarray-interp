use std::{fmt::Debug, ops::Sub};

use ndarray::{
    s, Array, ArrayBase, ArrayView, Axis, AxisDescription, Data, DimAdd, Dimension, IntoDimension,
    Ix1, NdIndex, OwnedRepr, RemoveAxis, Slice, Zip,
};
use num_traits::{Num, NumCast};
use thiserror::Error;

use crate::vector_extensions::{Monotonic, VectorExtensions};

mod strategies;
pub use self::strategies::*;

/// Errors during Interpolator creation
#[derive(Debug, Error)]
pub enum BuilderError {
    /// Insufficient data for the chosen interpolation strategy
    #[error("{0}")]
    NotEnoughData(String),
    /// A interpolation axis is not strict monotonic rising
    #[error("{0}")]
    Monotonic(String),
    /// The lengths of interpolation axis and the
    /// corresponding data axis do not match
    #[error("{0}")]
    AxisLenght(String),
    #[error("{0}")]
    DimensionError(String),
}

/// Errors during Interpolation
#[derive(Debug, Error)]
pub enum InterpolateError {
    #[error("{0}")]
    OutOfBounds(String),
}

/// One dimensional interpolator
#[derive(Debug)]
pub struct Interp1D<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + Debug,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
    Strat: Strategy<Sd, Sx, D>,
{
    /// x values are guaranteed to be strict monotonically rising
    /// if x is None, the x values are assumed to be the index of data
    x: Option<ArrayBase<Sx, Ix1>>,
    data: ArrayBase<Sd, D>,
    strategy: Strat,
    range: (Sx::Elem, Sx::Elem),
}

impl<Sd, Sx, Strat> Interp1D<Sd, Sx, Ix1, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Strat: Strategy<Sd, Sx, Ix1>,
{
    /// convinient interpolation function for interpolation at one point
    /// when the data dimension is [`type@Ix1`]
    ///
    /// ```rust
    /// # use ndarray_interp::*;
    /// # use ndarray::*;
    /// # use approx::*;
    /// let data = array![1.0, 1.5, 2.0];
    /// let x =    array![1.0, 2.0, 3.0];
    /// let query = 1.5;
    /// let expected = 1.25;
    ///
    /// let interpolator = Interp1DBuilder::new(data).x(x).build().unwrap();
    /// let result = interpolator.interp_scalar(query).unwrap();
    /// # assert_eq!(result, expected);
    /// ```
    pub fn interp_scalar(&self, x: Sx::Elem) -> Result<Sd::Elem, InterpolateError> {
        Ok(*self.interp(x)?.first().unwrap_or_else(|| unreachable!()))
    }
}

impl<Sd, D> Interp1D<Sd, OwnedRepr<Sd::Elem>, D, Linear>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug,
    D: Dimension + RemoveAxis,
{
    /// Get the [Interp1DBuilder]
    pub fn builder(data: ArrayBase<Sd, D>) -> Interp1DBuilder<Sd, OwnedRepr<Sd::Elem>, D, Linear> {
        Interp1DBuilder::new(data)
    }
}

impl<Sd, Sx, D, Strat> Interp1D<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    Strat: Strategy<Sd, Sx, D>,
{
    /// Calculate the interpolated values at `x`.
    /// Returns the interpolated data in an array one dimension smaller than
    /// the data dimension.
    ///
    /// ```rust
    /// # use ndarray_interp::*;
    /// # use ndarray::*;
    /// # use approx::*;
    /// // data has 2 dimension:
    /// let data = array![
    ///     [0.0, 2.0, 4.0],
    ///     [0.5, 2.5, 3.5],
    ///     [1.0, 3.0, 3.0],
    /// ];
    /// let query = 0.5;
    /// let expected = array![0.25, 2.25, 3.75];
    ///
    /// let interpolator = Interp1DBuilder::new(data).build().unwrap();
    /// let result = interpolator.interp(query).unwrap();
    /// # assert_abs_diff_eq!(result, expected, epsilon=f64::EPSILON);
    /// ```
    ///
    /// Concider using [`interp_scalar(x)`](Interp1D::interp_scalar)
    /// when the data dimension is [`type@Ix1`]
    pub fn interp(&self, x: Sx::Elem) -> Result<Array<Sd::Elem, D::Smaller>, InterpolateError> {
        let dim = self.data.raw_dim().remove_axis(Axis(0));
        let mut target: Array<Sd::Elem, _> = Array::zeros(dim);
        self.strategy
            .interp_into(self, target.view_mut(), x)
            .map(|_| target)
    }

    /// Calculate the interpolated values at all points in `xs`
    ///
    /// ```rust
    /// # use ndarray_interp::*;
    /// # use ndarray::*;
    /// # use approx::*;
    /// let data =     array![0.0,  0.5, 1.0 ];
    /// let x =        array![0.0,  1.0, 2.0 ];
    /// let query =    array![0.5,  1.0, 1.5 ];
    /// let expected = array![0.25, 0.5, 0.75];
    ///
    /// let interpolator = Interp1DBuilder::new(data)
    ///     .x(x)
    ///     .strategy(Linear{extrapolate: false})
    ///     .build().unwrap();
    /// let result = interpolator.interp_array(&query).unwrap();
    /// # assert_abs_diff_eq!(result, expected, epsilon=f64::EPSILON);
    /// ```
    ///
    /// # Dimensions
    /// given the data dimension is `N` and the dimension of `xs` is `M`
    /// the return array will have dimension `M + N - 1` where the first
    /// `M` dimensions correspond to the dimensions of `xs`.
    ///
    /// ```rust
    /// # use ndarray_interp::*;
    /// # use ndarray::*;
    /// # use approx::*;
    /// // data has 2 dimension:
    /// let data = array![
    ///     [0.0, 2.0],
    ///     [0.5, 2.5],
    ///     [1.0, 3.0],
    /// ];
    /// let x = array![
    ///     0.0,
    ///     1.0,
    ///     2.0,
    /// ];
    /// // query with 2 dimensions:
    /// let query = array![
    ///     [0.0, 0.5],
    ///     [1.0, 1.5],
    /// ];
    /// // expecting 3 dimensions!
    /// let expected = array![
    ///     [[0.0, 2.0], [0.25, 2.25]], // result for x=[0.0, 0.5]
    ///     [[0.5, 2.5], [0.75, 2.75]], // result for x=[1.0, 1.5]
    /// ];
    ///
    /// let interpolator = Interp1DBuilder::new(data)
    ///     .x(x)
    ///     .strategy(Linear{extrapolate: false})
    ///     .build().unwrap();
    /// let result = interpolator.interp_array(&query).unwrap();
    /// # assert_abs_diff_eq!(result, expected, epsilon=f64::EPSILON);
    /// ```
    pub fn interp_array<Dq>(
        &self,
        xs: &ArrayBase<Sx, Dq>,
    ) -> Result<Array<Sd::Elem, <Dq as DimAdd<D::Smaller>>::Output>, InterpolateError>
    where
        D: RemoveAxis,
        Dq: Dimension + DimAdd<D::Smaller>,
        Dq::Pattern: NdIndex<Dq>,
    {
        let mut dim = <Dq as DimAdd<D::Smaller>>::Output::default();
        dim.as_array_view_mut()
            .into_iter()
            .zip(
                xs.shape()
                    .iter()
                    .chain(self.data.raw_dim().as_array_view().slice(s![1..])),
            )
            .for_each(|(new_axis, len)| {
                *new_axis = *len;
            });
        let mut ys = Array::zeros(dim);

        // Perform interpolation for each index
        for (index, &x) in xs.indexed_iter() {
            let current_dim = index.clone().into_dimension();
            let subview =
                ys.slice_each_axis_mut(|AxisDescription { axis: Axis(nr), .. }| match current_dim
                    .as_array_view()
                    .get(nr)
                {
                    Some(idx) => Slice::from(*idx..*idx + 1),
                    None => Slice::from(..),
                });

            self.strategy.interp_into(
                self,
                subview
                    .into_shape(self.data.raw_dim().remove_axis(Axis(0)))
                    .unwrap_or_else(|_| unreachable!()),
                x,
            )?;
        }

        Ok(ys)
    }

    /// get x,data coordinate at given index
    /// panics at index out of range
    fn get_point(&self, idx: usize) -> (Sx::Elem, ArrayView<Sd::Elem, D::Smaller>) {
        let slice = Slice::from(idx..idx + 1);
        let view = self.data.slice_axis(Axis(0), slice).remove_axis(Axis(0));
        match &self.x {
            Some(x) => (*x.get(idx).unwrap_or_else(|| unreachable!()), view),
            None => (NumCast::from(idx).unwrap_or_else(|| unreachable!()), view),
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
/// # Default configuration
/// In the default configuration the interpolation strategy is [`Linear{extrapolate: false}`].
/// The data will be interpolated along [`Axis(0)`] (currently this can not be changed).
/// The index to `Axis(0)` of the data will be used as x values.
#[derive(Debug)]
pub struct Interp1DBuilder<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + Debug,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
{
    x: Option<ArrayBase<Sx, Ix1>>,
    data: ArrayBase<Sd, D>,
    strategy: Option<Strat>,
}

impl<Sd, D> Interp1DBuilder<Sd, OwnedRepr<Sd::Elem>, D, Linear>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug,
    D: Dimension,
{
    /// Create a new [Interp1DBuilder] and provide the data to interpolate.
    /// When nothing else is configured [Interp1DBuilder::build] will create an Interpolator using
    /// Linear Interpolation without extrapolation. As x axis the index to the data will be used.
    /// On multidimensional data interpolation happens along the first axis.
    pub fn new(data: ArrayBase<Sd, D>) -> Self {
        Interp1DBuilder {
            x: None,
            data,
            strategy: Some(Linear { extrapolate: false }),
        }
    }
}

impl<Sd, Sx, D, Strat> Interp1DBuilder<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
    Strat: StrategyBuilder<Sd, Sx, D>,
{
    /// Add an custom x axis for the data. The axis needs to have the same lenght
    /// and store the same Type as the data. `x`  must be strict monotonic rising.
    /// If the x axis is not set the index `0..data.len() - 1` is used
    pub fn x<NewSx>(self, x: ArrayBase<NewSx, Ix1>) -> Interp1DBuilder<Sd, NewSx, D, Strat>
    where
        NewSx: Data<Elem = Sd::Elem>,
    {
        let Interp1DBuilder { data, strategy, .. } = self;
        Interp1DBuilder {
            x: Some(x),
            data,
            strategy,
        }
    }

    /// Set the [Interp1DStrategy]. By default [Linear] with `Linear{extrapolate: false}` is used.
    pub fn strategy<NewStrat>(self, strategy: NewStrat) -> Interp1DBuilder<Sd, Sx, D, NewStrat>
    where
        NewStrat: StrategyBuilder<Sd, Sx, D>,
    {
        let Interp1DBuilder { x, data, .. } = self;
        Interp1DBuilder {
            x,
            data,
            strategy: Some(strategy),
        }
    }

    /// Validate input data and create the configured [Interp1D]
    pub fn build(mut self) -> Result<Interp1D<Sd, Sx, D, Strat::FinishedStrat>, BuilderError> {
        let &len = self
            .data
            .raw_dim()
            .as_array_view()
            .get(0)
            .ok_or(BuilderError::DimensionError("data dimension is 0".into()))?;
        if len < Strat::MINIMUM_DATA_LENGHT {
            return Err(BuilderError::NotEnoughData(format!(
                "The chosen Interpolation strategy needs at least {} data points",
                Strat::MINIMUM_DATA_LENGHT
            )));
        };

        if let Some(x) = &self.x {
            match x.monotonic_prop() {
                Monotonic::Rising { strict: true } => Ok(()),
                _ => Err(BuilderError::Monotonic(
                    "Values in the x axis need to be strictly monotonic rising".into(),
                )),
            }?;
            if *self
                .data
                .raw_dim()
                .as_array_view()
                .get(0)
                .unwrap_or_else(|| unreachable!())
                != x.len()
            {
                return Err(BuilderError::AxisLenght(format!(
                    "Lengths of x and data axis need to match. Got x: {:}, data: {:}",
                    x.len(),
                    self.data.len()
                )));
            };
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

        let strategy = self
            .strategy
            .take()
            .unwrap_or_else(|| {
                unreachable!("this is the only place where the option is set to None")
            })
            .build(&self)?;
        Ok(Interp1D {
            x: self.x,
            data: self.data,
            strategy: strategy,
            range,
        })
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use ndarray::s;
    use num_traits::NumCast;

    use super::Interp1D;
    use super::Interp1DBuilder;
    use crate::interp1d::strategies::Linear;
    use crate::BuilderError;
    use crate::InterpolateError;

    #[test]
    fn test_type_cast_assumptions() {
        assert_eq!(<i32 as NumCast>::from(1.75).unwrap(), 1);
        assert_eq!(<i32 as NumCast>::from(1.25).unwrap(), 1);
    }

    #[test]
    fn interp_y_only() {
        let interp = Interp1D::builder(array![1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.5])
            .build()
            .unwrap();
        assert_eq!(*interp.interp(0.0).unwrap().first().unwrap(), 1.5);
        assert_eq!(*interp.interp(9.0).unwrap().first().unwrap(), 10.5);
        assert_eq!(*interp.interp(4.5).unwrap().first().unwrap(), 6.0);
        assert_eq!(*interp.interp(0.25).unwrap().first().unwrap(), 1.625);
        assert_eq!(*interp.interp(8.75).unwrap().first().unwrap(), 10.125);
    }

    #[test]
    fn extrapolate_y_only() {
        let interp = Interp1D::builder(array![1.0, 2.0, 1.5])
            .strategy(Linear { extrapolate: true })
            .build()
            .unwrap();
        assert_eq!(*interp.interp(-1.0).unwrap().first().unwrap(), 0.0);
        assert_eq!(*interp.interp(3.0).unwrap().first().unwrap(), 1.0);
    }

    #[test]
    fn interp_with_x_and_y() {
        let interp =
            Interp1DBuilder::new(array![1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.5])
                .x(array![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                .strategy(Linear { extrapolate: false })
                .build()
                .unwrap();
        assert_eq!(*interp.interp(-4.0).unwrap().first().unwrap(), 1.5);
        assert_eq!(*interp.interp(5.0).unwrap().first().unwrap(), 10.5);
        assert_eq!(*interp.interp(0.5).unwrap().first().unwrap(), 6.0);
        assert_eq!(*interp.interp(-3.75).unwrap().first().unwrap(), 1.625);
        assert_eq!(*interp.interp(4.75).unwrap().first().unwrap(), 10.125);
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
        assert_eq!(*interp.interp(1.0).unwrap().first().unwrap(), 1.0);
        assert_eq!(*interp.interp(512.0).unwrap().first().unwrap(), 1.0);
        assert_eq!(*interp.interp(42.0).unwrap().first().unwrap(), 4.6875);
        assert_eq!(*interp.interp(365.0).unwrap().first().unwrap(), 1.57421875);
    }

    #[test]
    fn extrapolate_with_x_and_y() {
        let interp = Interp1DBuilder::new(array![1.0, 0.0, 1.5])
            .x(array![0.0, 1.0, 1.5])
            .strategy(Linear { extrapolate: true })
            .build()
            .unwrap();
        assert_eq!(*interp.interp(-1.0).unwrap().first().unwrap(), 2.0);
        assert_eq!(*interp.interp(2.0).unwrap().first().unwrap(), 3.0);
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
        let interp = Interp1D::builder(array![1.0, 2.0, 3.0]).build().unwrap();
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
            Interp1DBuilder::new(array![1]).build(),
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
        println!("{:?}", interp.interp(5.0).unwrap());
        assert_eq!(*interp.interp(-4.0).unwrap().first().unwrap(), 10.0);
        assert_eq!(*interp.interp(5.0).unwrap().first().unwrap(), 1.0);
        assert_eq!(*interp.interp(0.0).unwrap().first().unwrap(), 6.0);
        assert_eq!(*interp.interp(-3.5).unwrap().first().unwrap(), 9.5);
        assert_eq!(*interp.interp(4.75).unwrap().first().unwrap(), 1.25);
    }

    #[test]
    fn interp_multi_fn() {
        let data = array![
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [2.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [20.0, 40.0, 60.0, 80.0, 100.0],
        ];
        let interp = Interp1DBuilder::new(data)
            .x(array![1.0, 2.0, 3.0, 4.0])
            .build()
            .unwrap();
        let res = interp.interp(1.5).unwrap();
        assert_abs_diff_eq!(
            res,
            array![1.05, 1.1, 1.65, 2.2, 2.75],
            epsilon = f64::EPSILON
        );
        let array_array = interp
            .interp_array(&array![[1.0, 1.5], [3.5, 4.0]])
            .unwrap();

        assert_abs_diff_eq!(
            array_array.slice(s![1, 1, ..]),
            array![20.0, 40.0, 60.0, 80.0, 100.0],
            epsilon = f64::EPSILON
        );
        assert_abs_diff_eq!(
            array_array,
            array![
                [[0.1, 0.2, 0.3, 0.4, 0.5], [1.05, 1.1, 1.65, 2.2, 2.75]],
                [
                    [15.0, 30.0, 45.0, 60.0, 75.0],
                    [20.0, 40.0, 60.0, 80.0, 100.0]
                ]
            ],
            epsilon = f64::EPSILON
        );
    }
}
