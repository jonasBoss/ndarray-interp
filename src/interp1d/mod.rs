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
//!  - [`cubic_spline`] Cubic spline interpolation strategy

use std::{any::TypeId, fmt::Debug, ops::Sub};

use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, ArrayViewMut1, Axis, AxisDescription, Data, DimAdd,
    Dimension, IntoDimension, Ix1, OwnedRepr, RemoveAxis, Slice, Zip,
};
use num_traits::{cast, Num, NumCast};

use crate::{
    cast_unchecked,
    dim_extensions::DimExtension,
    vector_extensions::{Monotonic, VectorExtensions},
    BuilderError, InterpolateError,
};

mod aliases;
mod strategies;
pub use aliases::*;
pub use strategies::cubic_spline;
pub use strategies::linear::Linear;
pub use strategies::{Interp1DStrategy, Interp1DStrategyBuilder};

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
        let mut buffer: [Sd::Elem; 1] = [cast(0.0).unwrap_or_else(|| unimplemented!())];
        let buf_view = ArrayViewMut1::from(buffer.as_mut_slice()).remove_axis(Axis(0));
        self.strategy
            .interp_into(self, buf_view, x)
            .map(|_| buffer[0])
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

    /// Calculate the interpolated values at `x`.
    /// and stores the result into the provided buffer.
    ///
    /// The provided buffer must have the same shape as the interpolation data
    /// with the first axis removed.
    ///
    /// This can improve performance compared to [`interp`](Interp1D::interp)
    /// because it does not allocate any memory for the result
    ///
    /// # Panics
    /// When the provided buffer is too small or has the wrong shape
    pub fn interp_into(
        &self,
        x: Sx::Elem,
        buffer: ArrayViewMut<'_, Sd::Elem, D::Smaller>,
    ) -> Result<(), InterpolateError> {
        self.strategy.interp_into(self, buffer, x)
    }

    /// Calculate the interpolated values at all points in `xs`
    /// See [`interp_array_into`](Interp1D::interp_array_into) for dimension information
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
    pub fn interp_array<Sq, Dq>(
        &self,
        xs: &ArrayBase<Sq, Dq>,
    ) -> Result<Array<Sd::Elem, <Dq as DimAdd<D::Smaller>>::Output>, InterpolateError>
    where
        Sq: Data<Elem = Sd::Elem>,
        Dq: Dimension + DimAdd<D::Smaller> + 'static,
        <Dq as DimAdd<D::Smaller>>::Output: DimExtension,
    {
        let dim = self.get_buffer_shape(xs.raw_dim());
        debug_assert_eq!(dim.ndim(), self.data.ndim() + xs.ndim() - 1);

        let mut ys = Array::zeros(dim);
        self.interp_array_into(xs, ys.view_mut()).map(|_| ys)
    }

    /// Calculate the interpolated values at all points in `xs`
    /// and stores the result into the provided buffer
    ///
    /// This can improve performance compared to [`interp_array`](Interp1D::interp_array)
    /// because it does not allocate any memory for the result
    ///
    /// # Dimensions
    /// given the data dimension is `N` and the dimension of `xs` is `M`
    /// the buffer must have dimension `M + N - 1` where the first
    /// `M` dimensions correspond to the dimensions of `xs`.
    ///
    /// Lets assume we hava a data dimension of `N = (2, 3, 4)` and query this data
    /// with an array of dimension `M = (10)`, the return dimension will be `(10, 3, 4)`
    /// given a multi dimensional qurey of `M = (10, 20)` the return will be `(10, 20, 3, 4)`
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
    ///
    /// // we need 3 buffer dimensions
    /// let mut buffer = array![
    ///     [[0.0, 0.0], [0.0, 0.0]],
    ///     [[0.0, 0.0], [0.0, 0.0]],
    /// ];
    ///
    /// // what we expect in the buffer after interpolation
    /// let expected = array![
    ///     [[0.0, 2.0], [0.25, 2.25]], // result for x=[0.0, 0.5]
    ///     [[0.5, 2.5], [0.75, 2.75]], // result for x=[1.0, 1.5]
    /// ];
    ///
    /// let interpolator = Interp1DBuilder::new(data)
    ///     .x(x)
    ///     .strategy(Linear::new())
    ///     .build().unwrap();
    /// interpolator.interp_array_into(&query, buffer.view_mut()).unwrap();
    /// # assert_abs_diff_eq!(buffer, expected, epsilon=f64::EPSILON);
    /// ```
    ///
    /// # panics
    /// When the provided buffer is too small or has the wrong shape
    pub fn interp_array_into<Sq, Dq>(
        &self,
        xs: &ArrayBase<Sq, Dq>,
        mut buffer: ArrayViewMut<Sd::Elem, <Dq as DimAdd<D::Smaller>>::Output>,
    ) -> Result<(), InterpolateError>
    where
        Sq: Data<Elem = Sd::Elem>,
        Dq: Dimension + DimAdd<D::Smaller> + 'static,
        <Dq as DimAdd<D::Smaller>>::Output: DimExtension,
    {
        //self.dim_check(xs.raw_dim(), buffer.raw_dim());
        if TypeId::of::<Dq>() == TypeId::of::<Ix1>() {
            // Safety: We checked that `Dq` has type `Ix1`.
            //    Therefor the `&ArrayBase<Sq, Dq>` and `&ArrayBase<Sq, Ix1>` must be the same type.
            let xs_1d = unsafe { cast_unchecked::<&ArrayBase<Sq, Dq>, &ArrayBase<Sq, Ix1>>(xs) };
            // Safety: `<Dq as DimAdd<D::Smaller>>::Output>` reducees the dimension of `D` by one,
            //    and adds the dimension of `Dq`.
            //    Given that `Dq` has type `Ix1` the resulting dimension will be `D` again.
            //    `D` might be of type `IxDyn` In that case `IxDyn::Smaller` => `IxDyn` and also `Ix1::DimAdd<IxDyn>::Output` => `IxDyn`
            let buffer_d = unsafe {
                cast_unchecked::<
                    ArrayViewMut<Sd::Elem, <Dq as DimAdd<D::Smaller>>::Output>,
                    ArrayViewMut<Sd::Elem, D>,
                >(buffer)
            };
            return self.interp_array_into_1d(xs_1d, buffer_d);
        }

        // Perform interpolation for each index
        for (index, &x) in xs.indexed_iter() {
            let current_dim = index.clone().into_dimension();
            let subview =
                buffer.slice_each_axis_mut(|AxisDescription { axis: Axis(nr), .. }| {
                    match current_dim.as_array_view().get(nr) {
                        Some(idx) => Slice::from(*idx..*idx + 1),
                        None => Slice::from(..),
                    }
                });

            let subview = match subview.into_shape(self.data.raw_dim().remove_axis(Axis(0))) {
                Ok(view) => view,
                Err(err) => {
                    let expect = self.get_buffer_shape(xs.raw_dim()).into_pattern();
                    let got = buffer.dim();
                    panic!("{err} expected: {expect:?}, got: {got:?}")
                }
            };

            self.strategy.interp_into(self, subview, x)?;
        }
        Ok(())
    }

    fn interp_array_into_1d<Sq>(
        &self,
        xs: &ArrayBase<Sq, Ix1>,
        mut buffer: ArrayViewMut<'_, Sd::Elem, D>,
    ) -> Result<(), InterpolateError>
    where
        Sq: Data<Elem = Sd::Elem>,
    {
        Zip::from(xs)
            .and(buffer.axis_iter_mut(Axis(0)))
            .fold_while(Ok(()), |_, &x, buf| {
                match self.strategy.interp_into(self, buf, x) {
                    Ok(_) => ndarray::FoldWhile::Continue(Ok(())),
                    Err(e) => ndarray::FoldWhile::Done(Err(e)),
                }
            })
            .into_inner()
    }

    /// the required shape of the buffer when calling [`interp_array_into`]
    fn get_buffer_shape<Dq>(&self, dq: Dq) -> <Dq as DimAdd<D::Smaller>>::Output
    where
        Dq: Dimension + DimAdd<D::Smaller>,
        <Dq as DimAdd<D::Smaller>>::Output: DimExtension,
    {
        let binding = dq.as_array_view();
        let lenghts = binding.iter().chain(self.data.shape()[1..].iter()).copied();
        <Dq as DimAdd<D::Smaller>>::Output::new(lenghts)
    }

    /// Create a interpolator without any data validation. This is fast and cheap.
    ///
    /// # Safety
    /// The following data properties are assumed, but not checked:
    ///  - `x` is stricktly monotonic rising
    ///  - `data.shape()[0] == x.len()`
    ///  - the `strategy` is porperly initialized with the data
    pub fn new_unchecked(x: ArrayBase<Sx, Ix1>, data: ArrayBase<Sd, D>, strategy: Strat) -> Self {
        Interp1D { x, data, strategy }
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
            return Err(ShapeError(
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
            return Err(BuilderError::ShapeError(format!(
                "Lengths of x and data axis need to match. Got x: {:}, data: {:}",
                x.len(),
                data.shape()[0],
            )));
        }

        let strategy = strategy.build(&x, &data)?;

        Ok(Interp1D { x, data, strategy })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Array1, IxDyn};
    use rand::{
        distributions::{uniform::SampleUniform, Uniform},
        rngs::StdRng,
        Rng, SeedableRng,
    };

    use super::Interp1D;

    fn rand_arr<T: SampleUniform>(size: usize, range: (T, T), seed: u64) -> Array1<T> {
        Array::from_iter(
            StdRng::seed_from_u64(seed)
                .sample_iter(Uniform::new_inclusive(range.0, range.1))
                .take(size),
        )
    }

    macro_rules! get_interp {
        ($dim:expr, $shape:expr) => {{
            let arr = rand_arr(4usize.pow($dim), (0.0, 1.0), 64)
                .into_shape($shape)
                .unwrap();
            Interp1D::builder(arr).build().unwrap()
        }};
    }

    macro_rules! test_dim {
        ($name:ident, $dim:expr, $shape:expr) => {
            #[test]
            fn $name() {
                let interp = get_interp!($dim, $shape);
                let res = interp.interp(2.2).unwrap();
                assert_eq!(res.ndim(), $dim - 1);

                let mut buf = Array::zeros(res.dim());
                interp.interp_into(2.2, buf.view_mut()).unwrap();
                assert_abs_diff_eq!(buf, res, epsilon = f64::EPSILON);

                let query = array![[0.5, 1.0], [1.5, 2.0]];
                let res = interp.interp_array(&query).unwrap();
                assert_eq!(res.ndim(), $dim - 1 + query.ndim());

                let mut buf = Array::zeros(res.dim());
                interp.interp_array_into(&query, buf.view_mut()).unwrap();
                assert_abs_diff_eq!(buf, res, epsilon = f64::EPSILON);
            }
        };
    }

    test_dim!(interp1d_1d, 1, 4);
    test_dim!(interp1d_2d, 2, (4, 4));
    test_dim!(interp1d_3d, 3, (4, 4, 4));
    test_dim!(interp1d_4d, 4, (4, 4, 4, 4));
    test_dim!(interp1d_5d, 5, (4, 4, 4, 4, 4));
    test_dim!(interp1d_6d, 6, (4, 4, 4, 4, 4, 4));
    test_dim!(interp1d_7d, 7, IxDyn(&[4, 4, 4, 4, 4, 4, 4]));

    #[test]
    fn interp1d_1d_scalar() {
        let arr = rand_arr(4, (0.0, 1.0), 64);
        let _res: f64 = Interp1D::builder(arr) // type check f64 as return
            .build()
            .unwrap()
            .interp_scalar(2.2)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "expected: [4], got: [3]")]
    fn interp1d_2d_into_too_small() {
        let interp = get_interp!(2, (4, 4));
        let mut buf = Array::zeros(3);
        let _ = interp.interp_into(2.2, buf.view_mut());
    }

    #[test]
    #[should_panic(expected = "expected: [4], got: [5]")]
    fn interp1d_2d_into_too_big() {
        let interp = get_interp!(2, (4, 4));
        let mut buf = Array::zeros(5);
        let _ = interp.interp_into(2.2, buf.view_mut());
    }

    #[test]
    #[should_panic(expected = "expected: [2], got: [1]")] // this is not really a good message
    fn interp1d_2d_array_into_too_small1() {
        let arr = rand_arr((4usize).pow(2), (0.0, 1.0), 64)
            .into_shape((4, 4))
            .unwrap();
        let interp = Interp1D::builder(arr).build().unwrap();
        let mut buf = Array::zeros((1, 4));
        let _ = interp.interp_array_into(&array![2.2, 2.4], buf.view_mut());
    }

    #[test]
    #[should_panic]
    fn interp1d_2d_array_into_too_small2() {
        let arr = rand_arr((4usize).pow(2), (0.0, 1.0), 64)
            .into_shape((4, 4))
            .unwrap();
        let interp = Interp1D::builder(arr).build().unwrap();
        let mut buf = Array::zeros((2, 3));
        let _ = interp.interp_array_into(&array![2.2, 2.4], buf.view_mut());
    }

    #[test]
    #[should_panic]
    fn interp1d_2d_array_into_too_big1() {
        let arr = rand_arr((4usize).pow(2), (0.0, 1.0), 64)
            .into_shape((4, 4))
            .unwrap();
        let interp = Interp1D::builder(arr).build().unwrap();
        let mut buf = Array::zeros((3, 4));
        let _ = interp.interp_array_into(&array![2.2, 2.4], buf.view_mut());
    }

    #[test]
    #[should_panic]
    fn interp1d_2d_array_into_too_big2() {
        let arr = rand_arr((4usize).pow(2), (0.0, 1.0), 64)
            .into_shape((4, 4))
            .unwrap();
        let interp = Interp1D::builder(arr).build().unwrap();
        let mut buf = Array::zeros((2, 5));
        let _ = interp.interp_array_into(&array![2.2, 2.4], buf.view_mut());
    }
}
