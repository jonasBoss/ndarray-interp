use std::{fmt::Debug, ops::Sub};

use ndarray::{Data, Dimension, RemoveAxis, Zip};
use num_traits::{Num, NumCast};

use crate::{interp1d::Linear, InterpolateError};

use super::{Interp2DStrategy, Interp2DStrategyBuilder};

#[derive(Debug)]
pub struct Biliniar {
    extrapolate: bool,
}

impl<Sd, Sx, Sy, D> Interp2DStrategyBuilder<Sd, Sx, Sy, D> for Biliniar
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
{
    const MINIMUM_DATA_LENGHT: usize = 2;

    type FinishedStrat = Self;

    fn build(
        self,
        _x: &ndarray::ArrayBase<Sx, ndarray::Ix1>,
        _y: &ndarray::ArrayBase<Sy, ndarray::Ix1>,
        _data: &ndarray::ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, crate::BuilderError> {
        Ok(self)
    }
}

impl<Sd, Sx, Sy, D> Interp2DStrategy<Sd, Sx, Sy, D> for Biliniar
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
{
    fn interp_into(
        &self,
        interpolator: &crate::interp2d::Interp2D<Sd, Sx, Sy, D, Self>,
        target: ndarray::ArrayViewMut<'_, <Sd>::Elem, <D::Smaller as Dimension>::Smaller>,
        x: <Sx>::Elem,
        y: <Sy>::Elem,
    ) -> Result<(), crate::InterpolateError> {
        if !self.extrapolate && !interpolator.is_in_x_range(x) {
            return Err(InterpolateError::OutOfBounds(format!(
                "x = {x:?} is not in range"
            )));
        }
        if !self.extrapolate && !interpolator.is_in_y_range(y) {
            return Err(InterpolateError::OutOfBounds(format!(
                "y = {y:?} is not in range"
            )));
        }

        let (x_idx, y_idx) = interpolator.get_index_left_of(x, y);
        let (x1, y1, z11) = interpolator.index_point(x_idx, y_idx);
        let (_, _, z12) = interpolator.index_point(x_idx, y_idx + 1);
        let (_, _, z21) = interpolator.index_point(x_idx + 1, y_idx);
        let (x2, y2, z22) = interpolator.index_point(x_idx + 1, y_idx + 1);

        Zip::from(target)
            .and(z11)
            .and(z12)
            .and(z21)
            .and(z22)
            .for_each(|z, &z11, &z12, &z21, &z22| {
                let z1 = Linear::calc_frac((x1, z11), (x2, z21), x);
                let z2 = Linear::calc_frac((x1, z12), (x2, z22), x);
                *z = Linear::calc_frac((y1, z1), (y2, z2), y)
            });
        Ok(())
    }
}

impl Biliniar {
    pub fn new() -> Self {
        Biliniar { extrapolate: false }
    }

    pub fn extrapolate(mut self, yes: bool) -> Self {
        self.extrapolate = yes;
        self
    }
}

impl Default for Biliniar {
    fn default() -> Self {
        Self::new()
    }
}
