//! A collection of structs and traits to interpolate data along the first two axis
//!
//! # Interpolator
//!  - [`Interp2D`] The interpolator used with any strategy
//!  - [`Interp2DBuilder`] Configure the interpolator
//!
//! # Traits
//!  - [`Interp2DStrategy`] The trait used to specialize [`Interp2D`] with the correct strategy
//!  - [`Interp2DStrategyBuilder`] The trait used to specialize [`Interp2DBuilder`] to initialize the correct strategy
//!
//! # Strategies
//!  - [`Biliniar`] Linear interpolation strategy

use std::{fmt::Debug, ops::Sub};

use ndarray::{
    Array, Array1, ArrayBase, ArrayView, ArrayViewMut, Axis, AxisDescription, Data, DimAdd,
    Dimension, IntoDimension, Ix0, Ix1, Ix2, OwnedRepr, RemoveAxis, Slice,
};
use num_traits::{cast, Num, NumCast};

use crate::{
    dim_extensions::DimExtension,
    vector_extensions::{Monotonic, VectorExtensions},
    BuilderError, InterpolateError,
};

mod aliases;
mod strategies;
pub use aliases::*;
pub use strategies::{Biliniar, Interp2DStrategy, Interp2DStrategyBuilder};

/// Two dimensional interpolator
#[derive(Debug)]
pub struct Interp2D<Sd, Sx, Sy, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension,
{
    x: ArrayBase<Sx, Ix1>,
    y: ArrayBase<Sy, Ix1>,
    data: ArrayBase<Sd, D>,
    strategy: Strat,
}

impl<Sd, D> Interp2D<Sd, OwnedRepr<Sd::Elem>, OwnedRepr<Sd::Elem>, D, Biliniar>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    D: Dimension,
{
    /// Get the [Interp2DBuilder]
    pub fn builder(
        data: ArrayBase<Sd, D>,
    ) -> Interp2DBuilder<Sd, OwnedRepr<Sd::Elem>, OwnedRepr<Sd::Elem>, D, Biliniar> {
        Interp2DBuilder::new(data)
    }
}

impl<Sd, Sx, Sy, Strat> Interp2D<Sd, Sx, Sy, Ix2, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    Strat: Interp2DStrategy<Sd, Sx, Sy, Ix2>,
{
    /// convinient interpolation function for interpolation at one point
    /// when the data dimension is [`type@Ix2`]
    ///
    /// ```rust
    /// # use ndarray_interp::*;
    /// # use ndarray_interp::interp2d::*;
    /// # use ndarray::*;
    /// # use approx::*;
    /// let data = array![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ];
    /// let (qx, qy) = (0.0, 0.5);
    /// let expected = 1.5;
    ///
    /// let interpolator = Interp2D::builder(data).build().unwrap();
    /// let result = interpolator.interp_scalar(qx, qy).unwrap();
    /// # assert_eq!(result, expected);
    /// ```
    pub fn interp_scalar(&self, x: Sx::Elem, y: Sy::Elem) -> Result<Sd::Elem, InterpolateError> {
        let mut buffer = Array::zeros(Ix0());
        self.strategy
            .interp_into(self, buffer.view_mut(), x, y)
            .map(|_| buffer.first().unwrap_or_else(|| unreachable!()))
            .copied()
    }
}

impl<Sd, Sx, Sy, D, Strat> Interp2D<Sd, Sx, Sy, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
    Strat: Interp2DStrategy<Sd, Sx, Sy, D>,
{
    /// Calculate the interpolated values at `(x, y)`.
    /// Returns the interpolated data in an array two dimensions smaller than
    /// the data dimension.
    ///
    /// Concider using [`interp_scalar(x, y)`](Interp2D::interp_scalar)
    /// when the data dimension is [`type@Ix2`]
    pub fn interp(
        &self,
        x: Sx::Elem,
        y: Sy::Elem,
    ) -> Result<Array<Sd::Elem, <D::Smaller as Dimension>::Smaller>, InterpolateError> {
        let dim = self
            .data
            .raw_dim()
            .remove_axis(Axis(0))
            .remove_axis(Axis(0));
        let mut target = Array::zeros(dim);
        self.strategy
            .interp_into(self, target.view_mut(), x, y)
            .map(|_| target)
    }

    /// Calculate the interpolated values at `(x, y)`.
    /// and stores the result into the provided buffer
    ///
    /// The provided buffer must have the same shape as the interpolation data
    /// with the first two axes removed.
    ///
    /// This can improve performance compared to [`interp`](Interp2D::interp)
    /// because it does not allocate any memory for the result
    ///
    /// # Panics
    /// When the buffer does not have the correct shape.
    #[inline]
    pub fn interp_into(
        &self,
        x: Sx::Elem,
        y: Sy::Elem,
        buffer: ArrayViewMut<'_, Sd::Elem, <D::Smaller as Dimension>::Smaller>,
    ) -> Result<(), InterpolateError> {
        self.strategy.interp_into(self, buffer, x, y)
    }

    /// Calculate the interpolated values at all points in `(xs, ys)`
    ///
    /// See [`interp_array_into`](Interp2D::interp_array_into) for dimension information
    ///
    /// # panics
    /// when `xs.shape() != ys.shape()`
    pub fn interp_array<Sqx, Sqy, Dq>(
        &self,
        xs: &ArrayBase<Sqx, Dq>,
        ys: &ArrayBase<Sqy, Dq>,
    ) -> Result<
        Array<Sd::Elem, <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output>,
        InterpolateError,
    >
    where
        Sqx: Data<Elem = Sd::Elem>,
        Sqy: Data<Elem = Sy::Elem>,
        Dq: Dimension + DimAdd<<D::Smaller as Dimension>::Smaller>,
        <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output: DimExtension,
    {
        assert!(
            xs.shape() == ys.shape(),
            "`xs.shape()` and `ys.shape()` do not match"
        );
        let dim = self.get_buffer_shape(xs.raw_dim());
        let mut zs = Array::zeros(dim);
        self.interp_array_into(xs, ys, zs.view_mut()).map(|_| zs)
    }

    /// Calculate the interpolated values at all points in `(xs, ys)`
    /// and stores the result into the provided buffer
    ///
    /// This can improve performance compared to [`interp_array`](Interp2D::interp_array)
    /// because it does not allocate any memory for the result
    ///
    /// # Dimensions
    /// given the data dimension `N` and the query dimension `M` the return array
    /// will have the dimension `M + N - 2` where the fist `M` dimensions correspond
    /// to the query dimenions of `xs` and `ys`
    ///
    /// Lets assume we hava a data dimension of `N = (2, 3, 4, 5)` and query this data
    /// with an array of dimension `M = (10)`, the return dimension will be `(10, 4, 5)`
    /// given a multi dimensional qurey of `M = (10, 20)` the return will be `(10, 20, 4, 5)`
    ///
    /// # panics
    /// when `xs.shape() != ys.shape()` or when the buffer does not have the correct shape.
    pub fn interp_array_into<Sqx, Sqy, Dq>(
        &self,
        xs: &ArrayBase<Sqx, Dq>,
        ys: &ArrayBase<Sqy, Dq>,
        mut buffer: ArrayViewMut<
            '_,
            Sd::Elem,
            <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output,
        >,
    ) -> Result<(), InterpolateError>
    where
        Sqx: Data<Elem = Sd::Elem>,
        Sqy: Data<Elem = Sy::Elem>,
        Dq: Dimension + DimAdd<<D::Smaller as Dimension>::Smaller>,
        <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output: DimExtension,
    {
        assert!(
            xs.shape() == ys.shape(),
            "`xs.shape()` and `ys.shape()` do not match"
        );
        for (index, &x) in xs.indexed_iter() {
            let current_dim = index.clone().into_dimension();
            let y = *ys
                .get(current_dim.clone())
                .unwrap_or_else(|| unreachable!());
            let subview =
                buffer.slice_each_axis_mut(|AxisDescription { axis: Axis(nr), .. }| {
                    match current_dim.as_array_view().get(nr) {
                        Some(idx) => Slice::from(*idx..*idx + 1),
                        None => Slice::from(..),
                    }
                });

            let subview = match subview.into_shape(
                self.data
                    .raw_dim()
                    .remove_axis(Axis(0))
                    .remove_axis(Axis(0)),
            ) {
                Ok(view) => view,
                Err(err) => {
                    let expect = self.get_buffer_shape(xs.raw_dim()).into_pattern();
                    let got = buffer.dim();
                    panic!("{err} expected: {expect:?}, got: {got:?}")
                }
            };

            self.strategy.interp_into(self, subview, x, y)?;
        }
        Ok(())
    }

    /// the required shape of the buffer when calling [`interp_array_into`]
    fn get_buffer_shape<Dq>(
        &self,
        dq: Dq,
    ) -> <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output
    where
        Dq: Dimension + DimAdd<<D::Smaller as Dimension>::Smaller>,
        <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output: DimExtension,
    {
        let binding = dq.as_array_view();
        let lenghts = binding.iter().chain(self.data.shape()[2..].iter()).copied();
        <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output::new(lenghts)
    }

    /// Create a interpolator without any data validation. This is fast and cheap.
    ///
    /// # Safety
    /// The following data properties are assumed, but not checked:
    ///  - `x` and `y` are stricktly monotonic rising
    ///  - `data.shape()[0] == x.len()`, `data.shape()[1] == y.len()`
    ///  - the `strategy` is porperly initialized with the data
    pub fn new_unchecked(
        x: ArrayBase<Sx, Ix1>,
        y: ArrayBase<Sy, Ix1>,
        data: ArrayBase<Sd, D>,
        strategy: Strat,
    ) -> Self {
        Interp2D {
            x,
            y,
            data,
            strategy,
        }
    }

    /// get `(x, y, data)` coordinate at the given index
    ///
    /// # panics
    /// when index out of bounds
    pub fn index_point(
        &self,
        x_idx: usize,
        y_idx: usize,
    ) -> (
        Sx::Elem,
        Sx::Elem,
        ArrayView<Sd::Elem, <D::Smaller as Dimension>::Smaller>,
    ) {
        (
            self.x[x_idx],
            self.y[y_idx],
            self.data
                .index_axis(Axis(0), x_idx)
                .index_axis_move(Axis(0), y_idx),
        )
    }

    /// The index of a known value left of, or at x and y.
    ///
    /// This will never return the right most index,
    /// so calling [`index_point(x_idx+1, y_idx+1)`](Interp2D::index_point) is always safe.
    pub fn get_index_left_of(&self, x: Sx::Elem, y: Sy::Elem) -> (usize, usize) {
        (self.x.get_lower_index(x), self.y.get_lower_index(y))
    }

    pub fn is_in_x_range(&self, x: Sx::Elem) -> bool {
        self.x[0] <= x && x <= self.x[self.x.len() - 1]
    }
    pub fn is_in_y_range(&self, y: Sy::Elem) -> bool {
        self.y[0] <= y && y <= self.y[self.y.len() - 1]
    }
}

/// Create and configure a [Interp2D] interpolator.
#[derive(Debug)]
pub struct Interp2DBuilder<Sd, Sx, Sy, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension,
{
    x: ArrayBase<Sx, Ix1>,
    y: ArrayBase<Sy, Ix1>,
    data: ArrayBase<Sd, D>,
    strategy: Strat,
}

impl<Sd, D> Interp2DBuilder<Sd, OwnedRepr<Sd::Elem>, OwnedRepr<Sd::Elem>, D, Biliniar>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    D: Dimension,
{
    pub fn new(data: ArrayBase<Sd, D>) -> Self {
        let x = Array1::from_iter((0..data.shape()[0]).map(|i| {
            cast(i).unwrap_or_else(|| {
                unimplemented!("casting from usize to a number should always work")
            })
        }));
        let y = Array1::from_iter((0..data.shape()[1]).map(|i| {
            cast(i).unwrap_or_else(|| {
                unimplemented!("casting from usize to a number should always work")
            })
        }));
        Interp2DBuilder {
            x,
            y,
            data,
            strategy: Biliniar::new(),
        }
    }
}

impl<Sd, Sx, Sy, D, Strat> Interp2DBuilder<Sd, Sx, Sy, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
    Strat: Interp2DStrategyBuilder<Sd, Sx, Sy, D>,
{
    /// Set the interpolation strategy by provideing a [`Interp2DStrategyBuilder`].
    /// By default [`Biliniar`] is used.
    pub fn strategy<NewStrat: Interp2DStrategyBuilder<Sd, Sx, Sy, D>>(
        self,
        strategy: NewStrat,
    ) -> Interp2DBuilder<Sd, Sx, Sy, D, NewStrat> {
        let Interp2DBuilder { x, y, data, .. } = self;
        Interp2DBuilder {
            x,
            y,
            data,
            strategy,
        }
    }

    /// Add an custom x axis for the data.
    /// The axis must have the same lenght as the first axis of the data.
    pub fn x<NewSx: Data<Elem = Sd::Elem>>(
        self,
        x: ArrayBase<NewSx, Ix1>,
    ) -> Interp2DBuilder<Sd, NewSx, Sy, D, Strat> {
        let Interp2DBuilder {
            y, data, strategy, ..
        } = self;
        Interp2DBuilder {
            x,
            y,
            data,
            strategy,
        }
    }

    /// Add an custom y axis for the data.
    /// The axis must have the same lenght as the second axis of the data.
    pub fn y<NewSy: Data<Elem = Sd::Elem>>(
        self,
        y: ArrayBase<NewSy, Ix1>,
    ) -> Interp2DBuilder<Sd, Sx, NewSy, D, Strat> {
        let Interp2DBuilder {
            x, data, strategy, ..
        } = self;
        Interp2DBuilder {
            x,
            y,
            data,
            strategy,
        }
    }

    /// Validate the input and create the configured [`Interp2D`]
    pub fn build(self) -> Result<Interp2D<Sd, Sx, Sy, D, Strat::FinishedStrat>, BuilderError> {
        use self::Monotonic::*;
        use BuilderError::*;
        let Interp2DBuilder {
            x,
            y,
            data,
            strategy: stratgy_builder,
        } = self;
        if data.ndim() < 2 {
            return Err(DimensionError(
                "data dimension needs to be at least 2".into(),
            ));
        }
        if data.shape()[0] < Strat::MINIMUM_DATA_LENGHT {
            return Err(NotEnoughData(format!("The 0-dimension has not enough data for the chosen interpolation strategy. Provided: {}, Reqired: {}", data.shape()[0], Strat::MINIMUM_DATA_LENGHT)));
        }
        if data.shape()[1] < Strat::MINIMUM_DATA_LENGHT {
            return Err(NotEnoughData(format!("The 1-dimension has not enough data for the chosen interpolation strategy. Provided: {}, Reqired: {}", data.shape()[1], Strat::MINIMUM_DATA_LENGHT)));
        }
        if x.len() != data.shape()[0] {
            return Err(AxisLenght(format!(
                "Lenghts of x-axis and data-0-axis need to match. Got x: {}, data-0: {}",
                x.len(),
                data.shape()[0]
            )));
        }
        if y.len() != data.shape()[1] {
            return Err(AxisLenght(format!(
                "Lenghts of y-axis and data-1-axis need to match. Got y: {}, data-1: {}",
                y.len(),
                data.shape()[1]
            )));
        }
        if !matches!(x.monotonic_prop(), Rising { strict: true }) {
            return Err(Monotonic(
                "The x-axis needs to be strictly monotonic rising".into(),
            ));
        }
        if !matches!(y.monotonic_prop(), Rising { strict: true }) {
            return Err(Monotonic(
                "The y-axis needs to be strictly monotonic rising".into(),
            ));
        }

        let strategy = stratgy_builder.build(&x, &y, &data)?;
        Ok(Interp2D {
            x,
            y,
            data,
            strategy,
        })
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

    use super::Interp2D;

    fn rand_arr<T: SampleUniform>(size: usize, range: (T, T), seed: u64) -> Array1<T> {
        Array::from_iter(
            StdRng::seed_from_u64(seed)
                .sample_iter(Uniform::new_inclusive(range.0, range.1))
                .take(size),
        )
    }

    macro_rules! test_dim {
        ($name:ident, $dim:expr, $shape:expr) => {
            #[test]
            fn $name() {
                let arr = rand_arr(4usize.pow($dim), (0.0, 1.0), 64)
                    .into_shape($shape)
                    .unwrap();
                let interp = Interp2D::builder(arr).build().unwrap();
                let res = interp.interp(2.2, 2.2).unwrap();
                assert_eq!(res.ndim(), $dim - 2);

                let mut buf = Array::zeros(res.dim());
                interp.interp_into(2.2, 2.2, buf.view_mut()).unwrap();
                assert_abs_diff_eq!(buf, res, epsilon = f64::EPSILON);

                let x_query = array![[0.5, 1.0], [1.5, 2.0]];
                let y_query = array![[1.5, 2.0], [2.5, 3.0]];
                let res = interp.interp_array(&x_query, &y_query).unwrap();
                assert_eq!(res.ndim(), $dim - 2 + x_query.ndim());

                let mut buf = Array::zeros(res.dim());
                interp
                    .interp_array_into(&x_query, &y_query, buf.view_mut())
                    .unwrap();
                assert_abs_diff_eq!(buf, res, epsilon = f64::EPSILON);
            }
        };
    }

    test_dim!(interp2d_2d, 2, (4, 4));
    test_dim!(interp2d_3d, 3, (4, 4, 4));
    test_dim!(interp2d_4d, 4, (4, 4, 4, 4));
    test_dim!(interp2d_5d, 5, (4, 4, 4, 4, 4));
    test_dim!(interp2d_6d, 6, (4, 4, 4, 4, 4, 4));
    test_dim!(interp2d_7d, 7, IxDyn(&[4, 4, 4, 4, 4, 4, 4]));
    test_dim!(interp2d_8d, 8, IxDyn(&[4, 4, 4, 4, 4, 4, 4, 4]));

    #[test]
    fn interp2d_2d_scalar() {
        let arr = rand_arr(4usize.pow(2), (0.0, 1.0), 64)
            .into_shape((4, 4))
            .unwrap();
        let _res: f64 = Interp2D::builder(arr) // typecheck f64 as return type
            .build()
            .unwrap()
            .interp_scalar(2.2, 2.2)
            .unwrap();
    }
}
