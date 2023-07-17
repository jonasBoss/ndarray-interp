use std::{fmt::Debug, ops::Sub};

use ndarray::{Data, Dimension};
use num_traits::{Num, NumCast};

use super::{Strategy, StrategyBuilder};

pub struct Biliniar;

impl<Sd, Sx, Sy, D> StrategyBuilder<Sd, Sx, Sy, D> for Biliniar
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension,
{
    const MINIMUM_DATA_LENGHT: usize = 2;

    type FinishedStrat = Self;

    fn build(
        self,
        x: &ndarray::ArrayBase<Sx, ndarray::Ix1>,
        y: &ndarray::ArrayBase<Sy, ndarray::Ix1>,
        data: &ndarray::ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, crate::BuilderError> {
        todo!()
    }
}

impl<Sd, Sx, Sy, D> Strategy<Sd, Sx, Sy, D> for Biliniar
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension,
{
    fn interp_into(
        &self,
        interpolator: &crate::interp2d::Interp2D<Sd, Sx, Sy, D, Self>,
        target: ndarray::ArrayViewMut<'_, <Sd>::Elem, <D as Dimension>::Smaller>,
        x: <Sx>::Elem,
        y: <Sy>::Elem,
    ) -> Result<(), crate::InterpolateError> {
        todo!()
    }
}
