use std::{
    fmt::Debug,
    ops::{Add, Sub, SubAssign},
};

use ndarray::{
    s, Array, ArrayBase, ArrayViewMut, Axis, Data, Dimension, Ix1, RemoveAxis, ScalarOperand, Zip,
};
use num_traits::{cast, Num, NumCast, Pow};

use crate::{interp1d::Interp1D, BuilderError, InterpolateError};

use super::{Interp1DStrategy, Interp1DStrategyBuilder};

const AX0: Axis = Axis(0);

/// The CubicSpline 1d interpolation Strategy
///
/// # Example
/// From [Wikipedia](https://en.wikipedia.org/wiki/Spline_interpolation#Example)
/// ```
/// # use ndarray_interp::*;
/// # use ndarray_interp::interp1d::*;
/// # use ndarray::*;
/// # use approx::*;
///
/// let y = array![ 0.5, 0.0, 3.0];
/// let x = array![-1.0, 0.0, 3.0];
/// let query = Array::linspace(-1.0, 3.0, 10);
/// let interpolator = Interp1DBuilder::new(y)
///     .strategy(CubicSpline)
///     .x(x)
///     .build().unwrap();
///
/// let result = interpolator.interp_array(&query).unwrap();
/// let expect = array![
///     0.5,
///     0.2109053497942387,
///     0.020576131687242816,
///     0.01851851851851849,
///     0.21364883401920443,
///     0.5733882030178327,
///     1.0648148148148144,
///     1.6550068587105617,
///     2.3110425240054866,
///     3.0
/// ];
/// # assert_abs_diff_eq!(result, expect, epsilon=f64::EPSILON);
/// ```
#[derive(Debug)]
pub struct CubicSpline;
impl<Sd, Sx, D> Interp1DStrategyBuilder<Sd, Sx, D> for CubicSpline
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

    fn build<Sx2>(
        self,
        x: &ArrayBase<Sx2, Ix1>,
        data: &ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, BuilderError>
    where
        Sx2: Data<Elem = Sd::Elem>,
    {
        let (a, b) = self.calc_coefficients(x, data);
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

        /*
         * Calculate the coefficients c_a and c_b for the cubic spline the method is outlined on
         * https://en.wikipedia.org/wiki/Spline_interpolation#Example
         *
         * This requires solving the Linear equation A * k = rhs
         * The Thomas algorithm is used, because the matrix A will be tridiagonal and diagonally dominant.
         * (https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm)
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
                let x_left = x[0];
                let x_mid = x[1];
                let x_right = x[2];

                *a_up = one / (x_right - x_mid);
                *a_mid = two / (x_mid - x_left) + two / (x_right - x_mid);
                *a_low = one / (x_mid - x_left);
            });

        let x_0 = x[0];
        let x_1 = x[1];

        a_up[0] = one / (x_1 - x_0);
        a_mid[0] = two / (x_1 - x_0);

        // x_n and xn-1
        let x_n = x[len - 1];
        let x_n1 = x[len - 2];
        a_mid[len - 1] = two / (x_n - x_n1);
        a_low[len - 1] = one / (x_n - x_n1);

        // RHS vector
        let mut rhs: Array<Sd::Elem, D> = Array::zeros(dim.clone());

        for i in 1..len - 1 {
            let rhs = rhs.index_axis_mut(AX0, i);
            let y_left = data.index_axis(AX0, i - 1);
            let y_mid = data.index_axis(AX0, i);
            let y_right = data.index_axis(AX0, i + 1);
            let x_left = x[i - 1];
            let x_mid = x[i];
            let x_right = x[i + 1];
            Zip::from(y_left).and(y_mid).and(y_right).map_assign_into(
                rhs,
                |&y_left, &y_mid, &y_right| {
                    three
                        * ((y_mid - y_left) / (x_mid - x_left).pow(two)
                            + (y_right - y_mid) / (x_right - x_mid).pow(two))
                },
            );
        }

        let rhs_0 = rhs.index_axis_mut(AX0, 0);
        let data_0 = data.index_axis(AX0, 0);
        let data_1 = data.index_axis(AX0, 1);
        Zip::from(rhs_0)
            .and(data_0)
            .and(data_1)
            .for_each(|rhs_0, &y_0, &y_1| {
                *rhs_0 = three * (y_1 - y_0) / (x_1 - x_0).pow(two);
            });

        let rhs_n = rhs.index_axis_mut(AX0, len - 1);
        let data_n = data.index_axis(AX0, len - 1);
        let data_n1 = data.index_axis(AX0, len - 2);
        Zip::from(rhs_n)
            .and(data_n)
            .and(data_n1)
            .for_each(|rhs_n, &y_n, &y_n1| {
                *rhs_n = three * (y_n - y_n1) / (x_n - x_n1).pow(two);
            });

        // now solving With Thomas algorithm

        let mut rhs_left = rhs.index_axis(AX0, 0).into_owned();
        for i in 1..len {
            let w = a_low[i] / a_mid[i - 1];
            a_mid[i] -= w * a_up[i - 1];

            let rhs = rhs.index_axis_mut(AX0, i);
            Zip::from(rhs)
                .and(rhs_left.view_mut())
                .for_each(|rhs, rhs_left| {
                    let new_rhs = *rhs - w * *rhs_left;
                    *rhs = new_rhs;
                    *rhs_left = new_rhs;
                });
        }

        let mut k = Array::zeros(dim);
        Zip::from(k.index_axis_mut(AX0, len - 1))
            .and(rhs.index_axis(AX0, len - 1))
            .for_each(|k, &rhs| {
                *k = rhs / a_mid[len - 1];
            });

        let mut k_right = k.index_axis(AX0, len - 1).into_owned();
        for i in (0..len - 1).rev() {
            Zip::from(k.index_axis_mut(AX0, i))
                .and(k_right.view_mut())
                .and(rhs.index_axis(AX0, i))
                .for_each(|k, k_right, &rhs| {
                    let new_k = (rhs - a_up[i] * *k_right) / a_mid[i];
                    *k = new_k;
                    *k_right = new_k;
                })
        }

        let mut c_a = Array::zeros(a_b_dim.clone());
        let mut c_b = Array::zeros(a_b_dim);
        for index in 0..len - 1 {
            Zip::from(c_a.index_axis_mut(AX0, index))
                .and(c_b.index_axis_mut(AX0, index))
                .and(k.index_axis(AX0, index))
                .and(k.index_axis(AX0, index + 1))
                .and(data.index_axis(AX0, index))
                .and(data.index_axis(AX0, index + 1))
                .for_each(|c_a, c_b, &k, &k_right, &y, &y_right| {
                    *c_a = k * (x[index + 1] - x[index]) - (y_right - y);
                    *c_b = (y_right - y) - k_right * (x[index + 1] - x[index]);
                })
        }

        (c_a, c_b)
    }

    pub fn new() -> Self {
        Self
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

impl<Sd, Sx, D> Interp1DStrategy<Sd, Sx, D> for CubicSplineStrategy<Sd, D>
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
        if !interp.is_in_range(x) {
            return Err(InterpolateError::OutOfBounds(format!(
                "x = {x:#?} is not in range",
            )));
        }

        let idx = interp.get_index_left_of(x);
        let (x_left, data_left) = interp.index_point(idx);
        let (x_right, data_right) = interp.index_point(idx + 1);
        let a_left = self.a.index_axis(AX0, idx);
        let b_left = self.b.index_axis(AX0, idx);
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
