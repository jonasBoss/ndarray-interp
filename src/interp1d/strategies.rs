use std::{
    fmt::Debug,
    ops::{Add, Div, Sub},
};

use ndarray::{
    array, s, Array, Array1, ArrayBase, ArrayViewMut, AssignElem, Axis, Data, Dimension, Ix1,
    RemoveAxis, ScalarOperand, Slice, Zip,
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
        let mut idx = this.get_left_index(x);
        if idx == this.data.len() - 1 {
            idx -= 1;
        }

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
struct CubicSpline;
impl<Sd, Sx, D> StrategyBuilder<Sd, Sx, D> for CubicSpline
where
    Sd: Data,
    Sd::Elem: Debug
        + Num
        + Copy
        + PartialOrd
        + Sub
        + NumCast
        + Add
        + Pow<Sd::Elem, Output = Sd::Elem>
        + ScalarOperand,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    const MINIMUM_DATA_LENGHT: usize = 4;
    type FinishedStrat = CubicSplineStrategy<Sd, D>;

    fn build(
        self,
        builder: &Interp1DBuilder<Sd, Sx, D, Self>,
    ) -> Result<Self::FinishedStrat, BuilderError> {
        let dim = builder.data.raw_dim().clone();
        let len = *dim.as_array_view().get(0).unwrap_or_else(|| unreachable!());
        let (a, b) = match builder.x.as_ref() {
            Some(x) => self.calc_coefficients(x, &builder.data),
            None => {
                let x = Array::from_iter((0..len).into_iter().map(|n| {
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
        Sd::Elem:
            Num + Copy + Sub + NumCast + Add + Pow<Sd::Elem, Output = Sd::Elem> + ScalarOperand,
        Sx: Data<Elem = Sd::Elem>,
        D: Dimension + RemoveAxis,
    {
        let dim = data.raw_dim().clone();
        let len = dim[0];
        let mut a_b_dim = data.raw_dim().clone();
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
                let y_left = data
                    .slice_axis(Axis(0), Slice::from(0..0))
                    .remove_axis(Axis(0));
                let y_mid = data
                    .slice_axis(Axis(0), Slice::from(1..1))
                    .remove_axis(Axis(0));
                let y_right = data
                    .slice_axis(Axis(0), Slice::from(2..2))
                    .remove_axis(Axis(0));
                let x_left = x[0];
                let x_mid = x[1];
                let x_right = x[1];

                //bb.assign((y_m.sub(&y_l) / (x_m - x_l).pow(two) + y_r.sub(&y_m) / (x_r - x_m).pow(two)) * three);
                Zip::from(y_left)
                    .and(y_mid)
                    .and(y_right)
                    .map_assign_into(bb, |&y_left, &y_mid, &y_right| {
                        three
                            * ((y_mid - y_left) / (x_mid - x_left).pow(two)
                                + (y_right - y_mid) / (x_right - x_mid).pow(two))
                    }
                );
            }
        );
        
        let bb_0 = bb.index_axis_mut(Axis(0), 0);
        let data_0 = data.index_axis(Axis(0), 0);
        let data_1= data.index_axis(Axis(0), 0);
        Zip::from(bb_0)
            .and(data_0)
            .and(data_1)
            .for_each(|bb_0, &y_0, &y_1|{
                *bb_0 = three * (y_1 - y_0) / (x_1 - x_0).pow(two);
            });
        
        let bb_n = bb.index_axis_mut(Axis(0), len -1);
        let data_n = data.index_axis(Axis(0), len-1);
        let data_n1= data.index_axis(Axis(0), len-2);
        Zip::from(bb_n)
            .and(data_n)
            .and(data_n1)
            .for_each(|bb_n, &y_n, &y_n1|{
                *bb_n = three * (y_n - y_n1) / (x_n - x_n1).pow(two);
            });
        
        let mut c_star = Array::zeros(len);
        c_star[0] = a_up[0] / a_mid[0];
        for idx in 1..len{
            c_star[idx] = a_up[idx] / (a_mid[idx] - a_low[idx] * c_star[idx -1]);
        }

        let mut d_star = Array::zeros(dim.clone());
        let bb_0 = bb.index_axis(Axis(0), 0);
        Zip::from(d_star.index_axis_mut(Axis(0),0))
            .and(bb_0)
            .for_each(|d, &bb|{
                *d = bb/a_mid[0];
            });

        let mut d_star_left = d_star.index_axis(Axis(0), 0).into_owned();
        for idx in 1..len {
            d_star_left.assign(&d_star.index_axis(Axis(0), idx));
            Zip::from(d_star.index_axis_mut(Axis(0), idx))
                .and(d_star_left.view())
                .and(bb.index_axis(Axis(0), idx))
                .for_each(|d, &d_left, &bb|{
                    *d = (bb - a_low[idx] * d_left) / (a_mid[idx] - a_low[idx] * d_left);
                });
        }
        
        let mut xx = Array::zeros(dim);
        xx.index_axis_mut(Axis(0), len -1 ).assign(&d_star.index_axis(Axis(0), len-1));
        let mut xx_right = xx.index_axis(Axis(0), len-1).into_owned();

        for idx in (0..len-1).rev() {
            xx_right.assign(&xx.index_axis(Axis(0), idx));
            Zip::from(xx.index_axis_mut(Axis(0), idx))
                .and(xx_right.view())
                .and(d_star.index_axis(Axis(0), idx))
                .for_each(|xx, &xx_right, &d|{
                    *xx = d - c_star[idx]*xx_right;
                });
        }

        let mut a = Array::zeros(a_b_dim.clone());
        let mut b = Array::zeros(a_b_dim);
        for index in 0..len-1 {
            Zip::from(a.index_axis_mut(Axis(0), index))
            .and(b.index_axis_mut(Axis(0), index))
            .and(xx.index_axis(Axis(0), index))
            .and(xx.index_axis(Axis(0), index + 1))
            .and(data.index_axis(Axis(0), index))
            .and(data.index_axis(Axis(0), index+1))
            .for_each(|a,b,&xx,&xx_right,&y,&y_right|{
                *a = xx * (x[index+1]-x[index])-(y_right-y);
                *b = (y_right-y) - xx_right * (x[index+1]+x[index]);
            })
        }

        (a, b)
    }
}

#[derive(Debug)]
struct CubicSplineStrategy<Sd, D>
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
        interpolator: &Interp1D<Sd, Sx, D, Self>,
        target: ArrayViewMut<'_, <Sd>::Elem, <D as Dimension>::Smaller>,
        x: <Sx>::Elem,
    ) -> Result<(), InterpolateError> {
        todo!()
    }
}
