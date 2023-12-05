use std::{
    fmt::Debug,
    ops::{Add, Sub, SubAssign},
};

use ndarray::{
    s, Array, Array1, ArrayBase, ArrayViewMut, Axis, Data, Dimension, Ix1, RemoveAxis,
    ScalarOperand, Zip,
};
use num_traits::{cast, Num, NumCast, Pow};

use crate::{interp1d::Interp1D, BuilderError, InterpolateError};

use super::{Interp1DStrategy, Interp1DStrategyBuilder};

const AX0: Axis = Axis(0);

pub trait SplineNum:
    Debug
    + Num
    + Copy
    + PartialOrd
    + Sub
    + SubAssign
    + NumCast
    + Add
    + Pow<Self, Output = Self>
    + ScalarOperand
    + Send
{
}

impl<T> SplineNum for T where
    T: Debug
        + Num
        + Copy
        + PartialOrd
        + Sub
        + SubAssign
        + NumCast
        + Add
        + Pow<Self, Output = Self>
        + ScalarOperand
        + Send
{
}

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
///     .strategy(CubicSpline::new())
///     .x(x)
///     .build().unwrap();
///
/// let result = interpolator.interp_array(&query).unwrap();
/// let expect = array![
///     0.5,
///     0.1851851851851852,
///     0.01851851851851853,
///     -5.551115123125783e-17,
///     0.12962962962962965,
///     0.40740740740740755,
///     0.8333333333333331,
///     1.407407407407407,
///     2.1296296296296293, 3.0
/// ];
/// # assert_abs_diff_eq!(result, expect, epsilon=f64::EPSILON);
/// ```
#[derive(Debug)]
pub struct CubicSpline<T, D: Dimension> {
    extrapolate: bool,
    boundary: BoundaryCondition<T, D>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BoundaryCondition<T, D: Dimension> {
    Periodic,
    Natural,
    Clamped,
    NotAKnot,
    Individual(Array<RowBoundarys<T>, D>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum RowBoundarys<T> {
    Periodic,
    Mixed {
        left: SingleBoundary<T>,
        right: SingleBoundary<T>,
    },
}

impl<T: SplineNum> RowBoundarys<T> {
    pub const Natural: RowBoundarys<T> = RowBoundarys::Mixed {
        left: SingleBoundary::Natural,
        right: SingleBoundary::Natural,
    };
    pub const NotAKnot: RowBoundarys<T> = RowBoundarys::Mixed {
        left: SingleBoundary::NotAKnot,
        right: SingleBoundary::NotAKnot,
    };
    pub const Clamped: RowBoundarys<T> = RowBoundarys::Mixed {
        left: SingleBoundary::Clamped,
        right: SingleBoundary::Clamped,
    };
}

#[derive(Debug, PartialEq, Eq)]
pub enum SingleBoundary<T> {
    NotAKnot,
    Natural,
    Clamped,
    FirstDeriv(T),
    SecondDeriv(T),
}

impl<T: SplineNum> SingleBoundary<T> {
    fn specialize(&mut self) {
        match self {
            SingleBoundary::NotAKnot => (),
            SingleBoundary::Natural => {
                *self = Self::SecondDeriv(cast(0.0).unwrap_or_else(|| unimplemented!()))
            }
            SingleBoundary::Clamped => {
                *self = Self::FirstDeriv(cast(0.0).unwrap_or_else(|| unimplemented!()))
            }
            SingleBoundary::FirstDeriv(_) => (),
            SingleBoundary::SecondDeriv(_) => (),
        }
    }
}

impl<Sd, Sx, D> Interp1DStrategyBuilder<Sd, Sx, D> for CubicSpline<Sd::Elem, D>
where
    Sd: Data,
    Sd::Elem: SplineNum,
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
        let Self {
            extrapolate,
            boundary: _,
        } = self;
        let (a, b) = self.calc_coefficients(x, data);
        Ok(CubicSplineStrategy { a, b, extrapolate })
    }
}

impl<T, D> CubicSpline<T, D>
where
    D: Dimension + RemoveAxis,
    T: SplineNum,
{
    fn calc_coefficients<Sd, Sx>(
        self,
        x: &ArrayBase<Sx, Ix1>,
        data: &ArrayBase<Sd, D>,
    ) -> (Array<Sd::Elem, D>, Array<Sd::Elem, D>)
    where
        Sd: Data<Elem = T>,
        Sx: Data<Elem = T>,
    {
        let dim = data.raw_dim();
        let len = dim[0];

        let k: Array<T, D> = match self.boundary {
            BoundaryCondition::Periodic => todo!(),
            BoundaryCondition::Natural => Self::calc_k(x, data, RowBoundarys::Natural),
            BoundaryCondition::Clamped => Self::calc_k(x, data, RowBoundarys::Clamped),
            BoundaryCondition::NotAKnot => Self::calc_k(x, data, RowBoundarys::NotAKnot),
            BoundaryCondition::Individual(_bounds) => todo!(),
        };

        let mut a_b_dim = data.raw_dim();
        a_b_dim[0] -= 1;
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

    fn calc_k<Sd, Sx, _D>(
        x: &ArrayBase<Sx, Ix1>,
        data: &ArrayBase<Sd, _D>,
        boundary: RowBoundarys<T>,
    ) -> Array<T, _D>
    where
        _D: Dimension + RemoveAxis,
        Sd: Data<Elem = T>,
        Sx: Data<Elem = T>,
    {
        let dim = data.raw_dim();
        let len = dim[0];

        /*
         * Calculate the coefficients c_a and c_b for the cubic spline the method is outlined on
         * https://en.wikipedia.org/wiki/Spline_interpolation#Example
         *
         * This requires solving the Linear equation A * k = rhs
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
                let dxn = x[2] - x[1];
                let dxn_1 = x[1] - x[0];

                *a_up = dxn_1;
                *a_mid = two * (dxn + dxn_1);
                *a_low = dxn;
            });

        // RHS vector
        let mut rhs = Array::zeros(dim.clone());

        for i in 1..len - 1 {
            let rhs = rhs.index_axis_mut(AX0, i);
            let y_left = data.index_axis(AX0, i - 1);
            let y_mid = data.index_axis(AX0, i);
            let y_right = data.index_axis(AX0, i + 1);

            let dxn = x[i + 1] - x[i]; // dx(n)
            let dxn_1 = x[i] - x[i - 1]; // dx(n-1)

            Zip::from(y_left).and(y_mid).and(y_right).map_assign_into(
                rhs,
                |&y_left, &y_mid, &y_right| {
                    three * (dxn * (y_mid - y_left) / dxn_1 + dxn_1 * (y_right - y_mid) / dxn)
                },
            );
        }

        // apply boundary conditions
        match boundary {
            RowBoundarys::Periodic => todo!(),
            RowBoundarys::Mixed {
                left: SingleBoundary::Natural,
                right: SingleBoundary::Natural,
            } => {
                let x_0 = x[0];
                let x_1 = x[1];
                a_up[0] = x_1 - x_0;
                a_mid[0] = two * (x_1 - x_0);
                let rhs_0 = rhs.index_axis_mut(AX0, 0);
                let data_0 = data.index_axis(AX0, 0);
                let data_1 = data.index_axis(AX0, 1);
                Zip::from(rhs_0)
                    .and(data_0)
                    .and(data_1)
                    .for_each(|rhs_0, &y_0, &y_1| {
                        *rhs_0 = three * (y_1 - y_0);
                    });

                // x_n and xn-1
                let x_n = x[len - 1];
                let x_n1 = x[len - 2];
                a_mid[len - 1] = two * (x_n - x_n1);
                a_low[len - 1] = x_n - x_n1;
                let rhs_n = rhs.index_axis_mut(AX0, len - 1);
                let data_n = data.index_axis(AX0, len - 1);
                let data_n1 = data.index_axis(AX0, len - 2);
                Zip::from(rhs_n)
                    .and(data_n)
                    .and(data_n1)
                    .for_each(|rhs_n, &y_n, &y_n1| {
                        *rhs_n = three * (y_n - y_n1);
                    });
            }
            RowBoundarys::Mixed {
                left: SingleBoundary::NotAKnot,
                right: SingleBoundary::NotAKnot,
            } => {
                if len == 3 {
                    // We handle this case by constructing a parabola passing through given points.
                    let dx0 = x[1] - x[0];
                    let dx1 = x[2] - x[1];

                    let y0 = data.index_axis(AX0, 0);
                    let y1 = data.index_axis(AX0, 1);
                    let y2 = data.index_axis(AX0, 2);
                    let slope0 = (y1.to_owned() - y0) / dx0;
                    let slope1 = (y2.to_owned() - y1) / dx1;

                    a_mid[0] = one; // [0, 0]
                    a_up[0] = one; // [0, 1]
                    a_low[1] = dx1; // [1, 0]
                    a_mid[1] = two * (dx0 + dx1); // [1, 1]
                    a_up[1] = dx0; // [1, 2]
                    a_low[2] = one; // [2, 1]
                    a_mid[2] = one; // [2, 2]

                    rhs.index_axis_mut(AX0, 0).assign(&(&slope0 * two));
                    rhs.index_axis_mut(AX0, 1)
                        .assign(&((&slope1 * dx0 + &slope0 * dx1) * three));
                    rhs.index_axis_mut(AX0, 2).assign(&(slope1 * two));
                } else {
                    let dx0 = x[1] - x[0];
                    let dx1 = x[2] - x[1];
                    a_mid[0] = dx1;
                    let d = x[2] - x[0];
                    a_up[0] = d;
                    let tmp1 = (dx0 + two * d) * dx1;
                    Zip::from(rhs.index_axis_mut(AX0, 0))
                        .and(data.index_axis(AX0, 0))
                        .and(data.index_axis(AX0, 1))
                        .and(data.index_axis(AX0, 2))
                        .for_each(|b, &y0, &y1, &y2| {
                            *b = (tmp1 * (y1 - y0) / dx0 + dx0.pow(two) * (y2 - y1) / dx1) / d;
                        });

                    let dx_1 = x[len - 1] - x[len - 2];
                    let dx_2 = x[len - 2] - x[len - 3];
                    a_mid[len - 1] = dx_1;
                    let d = x[len - 1] - x[len - 3];
                    a_low[len - 1] = d;
                    let tmp1 = (two * d + dx_1) * dx_2;
                    Zip::from(rhs.index_axis_mut(AX0, len - 1))
                        .and(data.index_axis(AX0, len - 1))
                        .and(data.index_axis(AX0, len - 2))
                        .and(data.index_axis(AX0, len - 3))
                        .for_each(|b, &y_1, &y_2, &y_3| {
                            *b = (dx_1.pow(two) * (y_2 - y_3) / dx_2 + tmp1 * (y_1 - y_2) / dx_1)
                                / d;
                        });
                }
            }
            RowBoundarys::Mixed { left: _, right: _ } => todo!(),
        }
        Self::thomas(a_up, a_mid, a_low, rhs)
    }

    /// The Thomas algorithm is used, because the matrix A will be tridiagonal and diagonally dominant
    /// [https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm]
    fn thomas<_D>(
        a_up: Array1<T>,
        mut a_mid: Array1<T>,
        a_low: Array1<T>,
        mut rhs: Array<T, _D>,
    ) -> Array<T, _D>
    where
        _D: Dimension + RemoveAxis,
    {
        let dim = rhs.raw_dim();
        let len = dim[0];
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
        k
    }

    /// create a cubic-spline interpolation stratgy
    pub fn new() -> Self {
        Self {
            extrapolate: false,
            boundary: BoundaryCondition::NotAKnot,
        }
    }

    /// does the strategy extrapolate? Default is `false`
    pub fn extrapolate(mut self, extrapolate: bool) -> Self {
        self.extrapolate = extrapolate;
        self
    }

    /// set the boundary condition. default is [`BoundaryCondition::Natural`]
    pub fn boundary(mut self, boundary: BoundaryCondition<T, D>) -> Self {
        self.boundary = boundary;
        self
    }
}

impl<T, D> Default for CubicSpline<T, D>
where
    D: Dimension + RemoveAxis,
    T: SplineNum,
{
    fn default() -> Self {
        Self::new()
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
    extrapolate: bool,
}

impl<Sd, Sx, D> Interp1DStrategy<Sd, Sx, D> for CubicSplineStrategy<Sd, D>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub + Send,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    fn interp_into(
        &self,
        interp: &Interp1D<Sd, Sx, D, Self>,
        target: ArrayViewMut<'_, <Sd>::Elem, <D as Dimension>::Smaller>,
        x: <Sx>::Elem,
    ) -> Result<(), InterpolateError> {
        if !self.extrapolate && !interp.is_in_range(x) {
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
        Zip::from(data_left)
            .and(data_right)
            .and(a_left)
            .and(b_left)
            .and(target)
            .for_each(|&y_left, &y_right, &a_left, &b_left, y| {
                *y = (one - t) * y_left
                    + t * y_right
                    + t * (one - t) * (a_left * (one - t) + b_left * t);
            });
        Ok(())
    }
}
