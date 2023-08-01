//! A collection of structs and traits to interpolate data along the first axis
//!
//! # Interpolator
//!  - [`Interp1D`] The interpolator used with any strategy
//!  - [`Interp1DBuilder`] Configure the interpolator
//!
//! # Strategies
//!  - [`Linear`] Linear interpolation strategy
//!  - [`CubicSpline`] Cubic spline interpolation strategy

use std::{fmt::Debug, ops::Sub};

use ndarray::{
    Array, ArrayBase, ArrayView, Axis, AxisDescription, Data, DimAdd, Dimension, IntoDimension,
    Ix1, OwnedRepr, RemoveAxis, Slice,
};
use num_traits::{cast, Num, NumCast};

use crate::{
    vector_extensions::{Monotonic, VectorExtensions},
    BuilderError, InterpolateError,
};

mod strategies;
pub use strategies::{CubicSpline, Interp1DStrategy, Interp1DStrategyBuilder, Linear};

/// One dimensional interpolator
#[derive(Debug)]
pub struct Interp1D<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + Debug,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
    Strat: Interp1DStrategy<Sd, Sx, D>,
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
    Strat: Interp1DStrategy<Sd, Sx, Ix1>,
{
    /// convinient interpolation function for interpolation at one point
    /// when the data dimension is [`type@Ix1`]
    ///
    /// ```rust
    /// # use ndarray_interp::*;
    /// # use ndarray_interp::interp1d::*;
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
    Strat: Interp1DStrategy<Sd, Sx, D>,
{
    /// Calculate the interpolated values at `x`.
    /// Returns the interpolated data in an array one dimension smaller than
    /// the data dimension.
    ///
    /// ```rust
    /// # use ndarray_interp::*;
    /// # use ndarray_interp::interp1d::*;
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
    /// # use ndarray_interp::interp1d::*;
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
    /// # use ndarray_interp::interp1d::*;
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
    pub fn interp_array<Sq, Dq>(
        &self,
        xs: &ArrayBase<Sq, Dq>,
    ) -> Result<Array<Sd::Elem, <Dq as DimAdd<D::Smaller>>::Output>, InterpolateError>
    where
        Sq: Data<Elem = Sd::Elem>,
        Dq: Dimension + DimAdd<D::Smaller>,
    {
        let mut dim = <Dq as DimAdd<D::Smaller>>::Output::default();
        dim.as_array_view_mut()
            .into_iter()
            .zip(xs.shape().iter().chain(self.data.shape()[1..].iter()))
            .for_each(|(new_axis, &len)| {
                *new_axis = len;
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

    /// get `(x, data)` coordinate at given index
    ///
    /// # panics
    /// when index out of bounds
    pub fn index_point(&self, index: usize) -> (Sx::Elem, ArrayView<Sd::Elem, D::Smaller>) {
        let view = self.data.index_axis(Axis(0), index);
        match &self.x {
            Some(x) => (x[index], view),
            None => (NumCast::from(index).unwrap_or_else(|| unreachable!()), view),
        }
    }

    /// The index of a known value left of, or at x.
    ///
    /// This will never return the right most index,
    /// so calling [`index_point(idx+1)`](Interp1D::index_point) is always safe.
    pub fn get_index_left_of(&self, x: Sx::Elem) -> usize {
        if let Some(xs) = &self.x {
            xs.get_lower_index(x)
        } else if x < NumCast::from(0).unwrap_or_else(|| unimplemented!()) {
            0
        } else {
            // this relies on the fact that float -> int cast will return the next lower int
            // for positive values
            let x = NumCast::from(x)
                .unwrap_or_else(|| unimplemented!("x is positive, so this should always work"));
            if x >= self.data.shape()[0] - 1 {
                self.data.shape()[0] - 2
            } else {
                x
            }
        }
    }
}

/// Create and configure a [Interp1D] Interpolator.
/// 
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
    strategy: Strat,
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
            strategy: Linear { extrapolate: false },
        }
    }
}

impl<Sd, Sx, D, Strat> Interp1DBuilder<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
    Strat: Interp1DStrategyBuilder<Sd, Sx, D>,
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

    /// Set the interpolation strategy by providing a [Interp1DStrategyBuilder].
    /// By default [Linear] with `Linear{extrapolate: false}` is used.
    pub fn strategy<NewStrat>(self, strategy: NewStrat) -> Interp1DBuilder<Sd, Sx, D, NewStrat>
    where
        NewStrat: Interp1DStrategyBuilder<Sd, Sx, D>,
    {
        let Interp1DBuilder { x, data, .. } = self;
        Interp1DBuilder { x, data, strategy }
    }

    /// Validate input data and create the configured [Interp1D]
    pub fn build(self) -> Result<Interp1D<Sd, Sx, D, Strat::FinishedStrat>, BuilderError> {
        use self::Monotonic::*;
        use BuilderError::*;
        if self.data.ndim() < 1 {
            return Err(DimensionError(
                "data dimension is 0, needs to be at least 1".into(),
            ));
        }
        if self.data.shape()[0] < Strat::MINIMUM_DATA_LENGHT {
            return Err(NotEnoughData(format!(
                "The chosen Interpolation strategy needs at least {} data points",
                Strat::MINIMUM_DATA_LENGHT
            )));
        }

        if let Some(x) = &self.x {
            match x.monotonic_prop() {
                Rising { strict: true } => Ok(()),
                _ => Err(Monotonic(
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

        let strategy = match self.x.as_ref() {
            Some(x) => self.strategy.build(x, &self.data)?,
            None => {
                let len = self.data.raw_dim()[0];
                let x = Array::from_iter((0..len).map(|n| {
                    cast(n).unwrap_or_else(|| {
                        unimplemented!("casting from usize to a number should always work")
                    })
                }));
                self.strategy.build(&x, &self.data)?
            }
        };
        Ok(Interp1D {
            x: self.x,
            data: self.data,
            strategy,
            range,
        })
    }
}
