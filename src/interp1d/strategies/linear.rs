use std::{fmt::Debug, ops::Sub};

use ndarray::{ArrayBase, ArrayViewMut, Data, Dimension, Ix1, RemoveAxis, Zip};
use num_traits::{Num, NumCast};

use crate::{interp1d::Interp1D, BuilderError, InterpolateError};

use super::{Strategy, StrategyBuilder};

/// Linear Interpolation Strategy
#[derive(Debug)]
pub struct Linear {
    pub extrapolate: bool,
}

impl<Sd, Sx, D> StrategyBuilder<Sd, Sx, D> for Linear
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    const MINIMUM_DATA_LENGHT: usize = 2;
    type FinishedStrat = Linear;
    fn build<Sx2>(
        self,
        _x: &ArrayBase<Sx2, Ix1>,
        _data: &ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, BuilderError>
    where
        Sx2: Data<Elem = Sd::Elem>,
    {
        Ok(self)
    }
}

impl<Sd, Sx, D> Strategy<Sd, Sx, D> for Linear
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    fn interp_into(
        &self,
        interpolator: &Interp1D<Sd, Sx, D, Self>,
        target: ArrayViewMut<'_, <Sd>::Elem, <D as Dimension>::Smaller>,
        x: Sx::Elem,
    ) -> Result<(), InterpolateError> {
        let this = interpolator;
        if !self.extrapolate && !(this.range.0 <= x && x <= this.range.1) {
            return Err(InterpolateError::OutOfBounds(format!(
                "x = {x:#?} is not in range of {:#?}",
                this.range
            )));
        }

        // find the relevant index
        let idx = this.get_index_left_of(x);

        // lookup the data
        let (x1, y1) = this.index_point(idx);
        let (x2, y2) = this.index_point(idx + 1);

        // do interpolation
        Zip::from(target).and(y1).and(y2).for_each(|t, &y1, &y2| {
            *t = Self::calc_frac((x1, y1), (x2, y2), x);
        });
        Ok(())
    }
}

impl Linear {
    /// linearly interpolate/exrapolate between two points
    pub(crate) fn calc_frac<T>((x1, y1): (T, T), (x2, y2): (T, T), x: T) -> T
    where
        T: Num + Copy,
    {
        let b = y1;
        let m = (y2 - y1) / (x2 - x1);
        m * (x - x1) + b
    }
}
