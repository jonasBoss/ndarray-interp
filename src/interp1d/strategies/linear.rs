use std::{fmt::Debug, ops::Sub};

use ndarray::{ArrayViewMut, Data, Dimension, RemoveAxis, Zip};
use num_traits::{Num, NumCast};

use crate::{
    interp1d::{Interp1D, Interp1DBuilder},
    BuilderError, InterpolateError,
};

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
    fn build(
        self,
        _builder: &Interp1DBuilder<Sd, Sx, D, Self>,
    ) -> Result<Self::FinishedStrat, BuilderError> {
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
        let idx = this.get_left_index(x);

        // lookup the data
        let (x1, y1) = this.get_point(idx);
        let (x2, y2) = this.get_point(idx + 1);

        // do interpolation
        Zip::from(target).and(y1).and(y2).for_each(|t, &y1, &y2| {
            *t = Interp1D::<Sd, Sx, D, Self>::calc_frac((x1, y1), (x2, y2), x);
        });
        Ok(())
    }
}
