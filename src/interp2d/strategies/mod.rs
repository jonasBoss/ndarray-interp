use std::{fmt::Debug, ops::Sub};

use ndarray::{ArrayBase, ArrayViewMut, Data, Dimension, Ix1, RemoveAxis};
use num_traits::{Num, NumCast};

use crate::{BuilderError, InterpolateError};

use super::Interp2D;

mod bilinear;

pub use bilinear::Bilinear;

pub trait Interp2DStrategyBuilder<Sd, Sx, Sy, D>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
{
    const MINIMUM_DATA_LENGHT: usize;
    type FinishedStrat: Interp2DStrategy<Sd, Sx, Sy, D>;

    /// initialize the strategy by validating data and
    /// possibly calculating coefficients
    /// This method is called in [`Interp2DBuilder::build`](crate::interp2d::Interp2DBuilder::build)
    ///
    /// When this method is called by [`Interp2DBuilder`](crate::interp2d::Interp2DBuilder) the
    /// following properties are guaranteed:
    ///  - x and y is strictly monotonically rising
    ///  - the lenght of x equals the lenght of the data Axis 0
    ///  - the lenght of y equals the lenght of the data Axis 1
    ///  - the lenght is at least `MINIMUM_DATA_LENGHT`
    ///  - Interpolation in x will happen along Axis 0
    ///  - Interpolation in y will happen along Axis 1
    fn build(
        self,
        x: &ArrayBase<Sx, Ix1>,
        y: &ArrayBase<Sy, Ix1>,
        data: &ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, BuilderError>;
}

pub trait Interp2DStrategy<Sd, Sx, Sy, D>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
    Self: Sized,
{
    /// Interpolate the at position `(x, y)` into the target array.
    /// This is used internally by [`Interp2D`].
    ///
    /// When called by [`Interp2D`] the following
    /// properties are guaranteed:
    ///  - The shape of the target array matches the
    ///     shape of the data array (provided to the builder)
    ///     with the first two axes removed.
    ///  - x can be any valid `Sx::Elem`
    ///  - y cna be any valid `Sy::Elem`
    fn interp_into(
        &self,
        interpolator: &Interp2D<Sd, Sx, Sy, D, Self>,
        target: ArrayViewMut<'_, Sd::Elem, <D::Smaller as Dimension>::Smaller>,
        x: Sx::Elem,
        y: Sy::Elem,
    ) -> Result<(), InterpolateError>;
}
