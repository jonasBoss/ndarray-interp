use std::{fmt::Debug, ops::Sub};

use ndarray::{ArrayBase, ArrayViewMut, Data, Dimension, Ix1, RemoveAxis};
use num_traits::{Num, NumCast};

use crate::{BuilderError, InterpolateError};

use super::Interp2D;

mod biliniar;

pub use biliniar::Biliniar;

pub trait StrategyBuilder<Sd, Sx, Sy, D>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
{
    const MINIMUM_DATA_LENGHT: usize;
    type FinishedStrat: Strategy<Sd, Sx, Sy, D>;

    fn build(
        self,
        x: &ArrayBase<Sx, Ix1>,
        y: &ArrayBase<Sy, Ix1>,
        data: &ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, BuilderError>;
}

pub trait Strategy<Sd, Sx, Sy, D>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
    Self: Sized,
{
    fn interp_into(
        &self,
        interpolator: &Interp2D<Sd, Sx, Sy, D, Self>,
        target: ArrayViewMut<'_, Sd::Elem, <D::Smaller as Dimension>::Smaller>,
        x: Sx::Elem,
        y: Sy::Elem,
    ) -> Result<(), InterpolateError>;
}
