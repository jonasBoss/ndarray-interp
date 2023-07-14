use std::{
    fmt::Debug,
    ops::{Add, Sub, SubAssign},
};

use ndarray::{
    s, Array, ArrayBase, ArrayViewMut, Axis, Data, Dimension, Ix1, RemoveAxis, ScalarOperand,
    Slice, Zip,
};
use num_traits::{cast, Num, NumCast, Pow};

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

#[derive(Debug)]
pub struct CubicSpline;
impl<Sd, Sx, D> StrategyBuilder<Sd, Sx, D> for CubicSpline
where
    Sd: Data,
    Sd::Elem: Debug
        + Num
        + Copy
        + PartialOrd
        + Sub
        + SubAssign
        + NumCast
        + Add
        + Pow<Sd::Elem, Output = Sd::Elem>
        + ScalarOperand,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    const MINIMUM_DATA_LENGHT: usize = 3;
    type FinishedStrat = CubicSplineStrategy<Sd, D>;

    fn build(
        self,
        builder: &Interp1DBuilder<Sd, Sx, D, Self>,
    ) -> Result<Self::FinishedStrat, BuilderError> {
        let dim = builder.data.raw_dim();
        let len = *dim.as_array_view().get(0).unwrap_or_else(|| unreachable!());
        let (a, b) = match builder.x.as_ref() {
            Some(x) => self.calc_coefficients(x, &builder.data),
            None => {
                let x = Array::from_iter((0..len).map(|n| {
                    cast(n).unwrap_or_else(|| {
                        unimplemented!("casting from usize to a number should always work")
                    })
                }));
                self.calc_coefficients(&x, &builder.data)
            }
        };
        Ok(CubicSplineStrategy { a, b })
    }
}

impl CubicSpline {
    fn calc_coefficients<Sd, Sx, D>(
        self,
        x: &ArrayBase<Sx, Ix1>,
        data: &ArrayBase<Sd, D>,
    ) -> (Array<Sd::Elem, D>, Array<Sd::Elem, D>)
    where
        Sd: Data,
        Sd::Elem: Num
            + Copy
            + Sub
            + SubAssign
            + NumCast
            + Add
            + Pow<Sd::Elem, Output = Sd::Elem>
            + ScalarOperand
            + Debug,
        Sx: Data<Elem = Sd::Elem>,
        D: Dimension + RemoveAxis,
    {
        let dim = data.raw_dim();
        let len = dim[0];
        let mut a_b_dim = data.raw_dim();
        a_b_dim[0] -= 1;

        // we need to solve A x = b, where A is a matrix and b is a vector...

        /*
         * Solves Ax=B using the Thomas algorithm, because the matrix A will be tridiagonal and diagonally dominant.
         *
         * The method is outlined on the Wikipedia page for Tridiagonal Matrix Algorithm
         */

        // upper, middle and lower diagonal of A
        let mut a_up = Array::zeros(len);
        let mut a_mid = Array::zeros(len);
        let mut a_low = Array::zeros(len);

        let one: Sd::Elem = cast(1.0).unwrap_or_else(|| unimplemented!());
        let two: Sd::Elem = cast(2.0).unwrap_or_else(|| unimplemented!());
        let three: Sd::Elem = cast(3.0).unwrap_or_else(|| unimplemented!());

        Zip::from(a_up.slice_mut(s![1..-1]))
            .and(a_mid.slice_mut(s![1..-1]))
            .and(a_low.slice_mut(s![1..-1]))
            .and(x.windows(3))
            .for_each(|a_up, a_mid, a_low, x| {
                let x_left = *x.get(0).unwrap_or_else(|| unreachable!());
                let x_mid = *x.get(1).unwrap_or_else(|| unreachable!());
                let x_right = *x.get(2).unwrap_or_else(|| unreachable!());

                *a_up = one / (x_right - x_mid);
                *a_mid = two / (x_mid - x_left) + two / (x_right - x_mid);
                *a_low = one / (x_mid - x_left);
            });

        let x_0 = *x.get(0).unwrap_or_else(|| unreachable!());
        let x_1 = *x.get(1).unwrap_or_else(|| unreachable!());

        *a_up.first_mut().unwrap_or_else(|| unreachable!()) = one / (x_1 - x_0);
        *a_mid.first_mut().unwrap_or_else(|| unreachable!()) = two / (x_1 - x_0);

        // x_n and xn-1
        let x_n = *x.get(len - 1).unwrap_or_else(|| unreachable!());
        let x_n1 = *x.get(len - 2).unwrap_or_else(|| unreachable!());
        *a_mid.last_mut().unwrap_or_else(|| unreachable!()) = two / (x_n - x_n1);
        *a_low.last_mut().unwrap_or_else(|| unreachable!()) = one / (x_n - x_n1);

        // RHS vector
        let mut bb: Array<Sd::Elem, D> = Array::zeros(dim.clone());

        let mut inner_bb = bb.slice_axis_mut(Axis(0), Slice::from(1..-1));
        Zip::from(inner_bb.axis_iter_mut(Axis(0)))
            .and(x.windows(3))
            .and(data.axis_windows(Axis(0), 3))
            .for_each(|bb, x, data| {
                let y_left = data.index_axis(Axis(0), 0);
                let y_mid = data.index_axis(Axis(0), 1);
                let y_right = data.index_axis(Axis(0), 2);
                let x_left = x[0];
                let x_mid = x[1];
                let x_right = x[2];

                //bb.assign((y_m.sub(&y_l) / (x_m - x_l).pow(two) + y_r.sub(&y_m) / (x_r - x_m).pow(two)) * three);
                Zip::from(y_left).and(y_mid).and(y_right).map_assign_into(
                    bb,
                    |&y_left, &y_mid, &y_right| {
                        three
                            * ((y_mid - y_left) / (x_mid - x_left).pow(two)
                                + (y_right - y_mid) / (x_right - x_mid).pow(two))
                    },
                );
            });

        let bb_0 = bb.index_axis_mut(Axis(0), 0);
        let data_0 = data.index_axis(Axis(0), 0);
        let data_1 = data.index_axis(Axis(0), 1);
        Zip::from(bb_0)
            .and(data_0)
            .and(data_1)
            .for_each(|bb_0, &y_0, &y_1| {
                *bb_0 = three * (y_1 - y_0) / (x_1 - x_0).pow(two);
            });

        let bb_n = bb.index_axis_mut(Axis(0), len - 1);
        let data_n = data.index_axis(Axis(0), len - 1);
        let data_n1 = data.index_axis(Axis(0), len - 2);
        Zip::from(bb_n)
            .and(data_n)
            .and(data_n1)
            .for_each(|bb_n, &y_n, &y_n1| {
                *bb_n = three * (y_n - y_n1) / (x_n - x_n1).pow(two);
            });

        // now solving With Thomas algorithm

        let mut bb_left = bb.index_axis(Axis(0), 0).into_owned();
        for i in 1..len {
            let w = a_low[i] / a_mid[i - 1];
            a_mid[i] -= w * a_up[i - 1];

            let bb = bb.index_axis_mut(Axis(0), i);
            Zip::from(bb)
                .and(bb_left.view_mut())
                .for_each(|bb, bb_left| {
                    let new_bb = *bb - w * *bb_left;
                    *bb = new_bb;
                    *bb_left = new_bb;
                });
        }

        let mut xx = Array::zeros(dim);
        Zip::from(xx.index_axis_mut(Axis(0), len - 1))
            .and(bb.index_axis(Axis(0), len - 1))
            .for_each(|k, &bb| {
                *k = bb / a_mid[len - 1];
            });

        let mut k_right = xx.index_axis(Axis(0), len - 1).into_owned();
        for i in (0..len - 1).rev() {
            Zip::from(xx.index_axis_mut(Axis(0), i))
                .and(k_right.view_mut())
                .and(bb.index_axis(Axis(0), i))
                .for_each(|k, k_right, &bb| {
                    let new_k = (bb - a_up[i] * *k_right) / a_mid[i];
                    *k = new_k;
                    *k_right = new_k;
                })
        }

        let mut a = Array::zeros(a_b_dim.clone());
        let mut b = Array::zeros(a_b_dim);
        for index in 0..len - 1 {
            Zip::from(a.index_axis_mut(Axis(0), index))
                .and(b.index_axis_mut(Axis(0), index))
                .and(xx.index_axis(Axis(0), index))
                .and(xx.index_axis(Axis(0), index + 1))
                .and(data.index_axis(Axis(0), index))
                .and(data.index_axis(Axis(0), index + 1))
                .for_each(|a, b, &xx, &xx_right, &y, &y_right| {
                    *a = xx * (x[index + 1] - x[index]) - (y_right - y);
                    *b = (y_right - y) - xx_right * (x[index + 1] - x[index]);
                })
        }

        (a, b)
    }
}

#[derive(Debug)]
pub struct CubicSplineStrategy<Sd, D>
where
    Sd: Data,
    D: Dimension + RemoveAxis,
{
    a: Array<Sd::Elem, D>,
    b: Array<Sd::Elem, D>,
}

impl<Sd, Sx, D> Strategy<Sd, Sx, D> for CubicSplineStrategy<Sd, D>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    fn interp_into(
        &self,
        interp: &Interp1D<Sd, Sx, D, Self>,
        target: ArrayViewMut<'_, <Sd>::Elem, <D as Dimension>::Smaller>,
        x: <Sx>::Elem,
    ) -> Result<(), InterpolateError> {
        if !(interp.range.0 <= x && x <= interp.range.1) {
            return Err(InterpolateError::OutOfBounds(format!(
                "x = {x:#?} is not in range of {:#?}",
                interp.range
            )));
        }

        let idx = interp.get_left_index(x);
        let (x_left, data_left) = interp.get_point(idx);
        let (x_right, data_right) = interp.get_point(idx + 1);
        let a_left = self.a.index_axis(Axis(0), idx);
        let b_left = self.b.index_axis(Axis(0), idx);
        let one: Sd::Elem = cast(1.0).unwrap_or_else(|| unimplemented!());

        let t = (x - x_left) / (x_right - x_left);
        Zip::from(target)
            .and(data_left)
            .and(data_right)
            .and(a_left)
            .and(b_left)
            .for_each(|y, &y_left, &y_right, &a_left, &b_left| {
                *y = (one - t) * y_left
                    + t * y_right
                    + t * (one - t) * (a_left * (one - t) + b_left * t);
            });
        Ok(())
    }
}
