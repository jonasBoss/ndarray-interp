//! A collection of structs and traits to interpolate data along the first axis
//!
//! # Interpolator
//!  - [`Interp1D`] The interpolator used with any strategy
//!  - [`Interp1DBuilder`] Configure the interpolator
//!
//! # Traits
//!  - [`Interp1DStrategy`] The trait used to specialize [`Interp1D`] with the correct strategy
//!  - [`Interp1DStrategyBuilder`] The trait used to specialize [`Interp1DBuilder`] to initialize the correct strategy
//!
//! # Strategies
//!  - [`Linear`] Linear interpolation strategy
//!  - [`CubicSpline`] Cubic spline interpolation strategy

use std::{cell::RefCell, fmt::Debug, ops::Sub};

use ndarray::{
    Array, ArrayBase, ArrayView, Axis, AxisDescription, Data, DimAdd, Dimension, IntoDimension,
    Ix1, OwnedRepr, RemoveAxis, Slice,
};
use num_traits::{cast, Num, NumCast};
use thread_local::ThreadLocal;

use crate::{
    vector_extensions::{Monotonic, VectorExtensions},
    BuilderError, InterpolateError,
};

mod aliases;
mod strategies;
pub use aliases::*;
pub use strategies::{CubicSpline, Interp1DStrategy, Interp1DStrategyBuilder, Linear};

/// One dimensional interpolator
#[derive(Debug)]
pub struct Interp1D<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + Debug + Send,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
    Strat: Interp1DStrategy<Sd, Sx, D>,
{
    /// x values are guaranteed to be strict monotonically rising
    x: ArrayBase<Sx, Ix1>,
    data: ArrayBase<Sd, D>,
    strategy: Strat,

    /// a thread local buffer, used by interp_scalar to avoid heap allocations
    buffer: ThreadLocal<RefCell<Array<Sd::Elem, D::Smaller>>>,
}

impl<Sd, Sx, Strat> Interp1D<Sd, Sx, Ix1, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
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
        let mut buffer = self
            .buffer
            .get_or(|| {
                let dim = self.data.raw_dim().remove_axis(Axis(0));
                RefCell::new(Array::zeros(dim))
            })
            .borrow_mut();
        let mut target = buffer.view_mut();
        self.strategy
            .interp_into(self, target.view_mut(), x)
            .map(|_| target.first().unwrap_or_else(|| unreachable!()))
            .copied()
    }
}

impl<Sd, D> Interp1D<Sd, OwnedRepr<Sd::Elem>, D, Linear>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Send,
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
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    Strat: Interp1DStrategy<Sd, Sx, D>,
{
    /// Create a interpolator without any data validation. This is fast and cheap.
    ///
    /// # Safety
    /// The following data properties are assumed, but not checked:
    ///  - `x` is stricktly monotonic rising
    ///  - `data.shape()[0] == x.len()`
    ///  - the `strategy` is porperly initialized with the data
    pub fn new_unchecked(x: ArrayBase<Sx, Ix1>, data: ArrayBase<Sd, D>, strategy: Strat) -> Self {
        let buffer: ThreadLocal<RefCell<Array<Sd::Elem, D::Smaller>>> = ThreadLocal::new();
        Interp1D {
            x,
            data,
            strategy,
            buffer,
        }
    }

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
    ///     .strategy(Linear::new())
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
    ///     .strategy(Linear::new())
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
        (self.x[index], view)
    }

    /// The index of a known value left of, or at x.
    ///
    /// This will never return the right most index,
    /// so calling [`index_point(idx+1)`](Interp1D::index_point) is always safe.
    pub fn get_index_left_of(&self, x: Sx::Elem) -> usize {
        self.x.get_lower_index(x)
    }

    pub fn is_in_range(&self, x: Sx::Elem) -> bool {
        self.x[0] <= x && x <= self.x[self.x.len() - 1]
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
    x: ArrayBase<Sx, Ix1>,
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
        let len = data.shape()[0];
        Interp1DBuilder {
            x: Array::from_iter((0..len).map(|n| {
                cast(n).unwrap_or_else(|| {
                    unimplemented!("casting from usize to a number should always work")
                })
            })),
            data,
            strategy: Linear::new(),
        }
    }
}

impl<Sd, Sx, D, Strat> Interp1DBuilder<Sd, Sx, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Send,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
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
        Interp1DBuilder { x, data, strategy }
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

        let Interp1DBuilder { x, data, strategy } = self;

        if data.ndim() < 1 {
            return Err(DimensionError(
                "data dimension is 0, needs to be at least 1".into(),
            ));
        }
        if data.shape()[0] < Strat::MINIMUM_DATA_LENGHT {
            return Err(NotEnoughData(format!(
                "The chosen Interpolation strategy needs at least {} data points",
                Strat::MINIMUM_DATA_LENGHT
            )));
        }
        if !matches!(x.monotonic_prop(), Rising { strict: true }) {
            return Err(Monotonic(
                "Values in the x axis need to be strictly monotonic rising".into(),
            ));
        }
        if x.len() != data.shape()[0] {
            return Err(BuilderError::AxisLenght(format!(
                "Lengths of x and data axis need to match. Got x: {:}, data: {:}",
                x.len(),
                data.shape()[0],
            )));
        }

        let strategy = strategy.build(&x, &data)?;

        let buffer: ThreadLocal<RefCell<Array<Sd::Elem, D::Smaller>>> = ThreadLocal::new();
        Ok(Interp1D {
            x,
            data,
            strategy,
            buffer,
        })
    }
}
