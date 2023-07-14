use std::fmt::Debug;

use ndarray::{ArrayViewMut, Data, Dimension};
use num_traits::Num;

use super::{Interp1D, Interp1DBuilder};
use crate::{BuilderError, InterpolateError};

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
    fn build(
        self,
        builder: &Interp1DBuilder<Sd, Sx, D, Self>,
    ) -> Result<Self::FinishedStrat, BuilderError>;
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
    /// When usde outside of [`Interp1D`] the behaviour is
    /// undefined, possibly causing a panic.
    fn interp_into(
        &self,
        interpolator: &Interp1D<Sd, Sx, D, Self>,
        target: ArrayViewMut<'_, Sd::Elem, D::Smaller>,
        x: Sx::Elem,
    ) -> Result<(), InterpolateError>;
}

mod cubic_spline;
mod linear;

pub use cubic_spline::CubicSpline;
pub use linear::Linear;
