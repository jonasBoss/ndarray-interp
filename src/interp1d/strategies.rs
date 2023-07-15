use std::fmt::Debug;

use ndarray::{ArrayBase, ArrayViewMut, Data, Dimension, Ix1};
use num_traits::Num;

use super::Interp1D;
use crate::{BuilderError, InterpolateError};

mod cubic_spline;
mod linear;

pub use cubic_spline::CubicSpline;
pub use linear::Linear;

pub trait StrategyBuilder<Sd, Sx, D>
where
    Sd: Data,
    Sd::Elem: Num + Debug,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
    Self: Sized,
{
    const MINIMUM_DATA_LENGHT: usize;
    type FinishedStrat: Strategy<Sd, Sx, D>;

    /// initialize the strategy by validating data and
    /// possibly calculating coefficients
    /// This method is called by [`Interp1DBuilder::build`](crate::interp1d::Interp1DBuilder::build)
    ///
    /// When this method is called by [`Interp1DBuilder`](crate::interp1d::Interp1DBuilder) the
    /// following properties are guaranteed:
    ///  - x is strictly monotonically rising
    ///  - the lenght of x equals the lenght of the data Axis 0
    ///  - the lenght is at least `MINIMUM_DATA_LENGHT`
    ///  - Interpolation will happen along axis 0
    fn build<Sx2>(
        self,
        x: &ArrayBase<Sx2, Ix1>,
        data: &ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, BuilderError>
    where
        Sx2: Data<Elem = Sd::Elem>;
}

pub trait Strategy<Sd, Sx, D>
where
    Sd: Data,
    Sd::Elem: Num + Debug,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension,
    Self: Sized,
{
    /// Interpolate the at position x into the target array.
    /// This is used internally by [`Interp1D`].
    ///
    /// When called by [`Interp1D`] the following
    /// properties are guaranteed:
    ///  - The shape of the target array matches the
    ///     shape of the data array (provided to the builder)
    ///     with the first axis removed.
    ///  - x can be any valid `Sx::Elem`
    fn interp_into(
        &self,
        interpolator: &Interp1D<Sd, Sx, D, Self>,
        target: ArrayViewMut<'_, Sd::Elem, D::Smaller>,
        x: Sx::Elem,
    ) -> Result<(), InterpolateError>;
}
