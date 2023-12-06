use std::{
    fmt::Debug,
    ops::{Add, Sub, SubAssign},
};

use ndarray::{
    s, Array, Array1, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension, Ix1, IxDyn,
    RemoveAxis, ScalarOperand, Zip,
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
///  # use ndarray_interp::interp1d::cubic_spline::*;
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

/// Boundary conditions for the whole dataset
///
/// The boundary condition is structured in three hirarchic enum's:
///  - This: The toplevel boundary applys to the whole dataset
///  - [`RowBoundary`] applys to a single row in the dataset
///  - [`SingleBoundary`] applys to an individual boundary of a single row
///
/// the default is the [`NotAKnot`](BoundaryCondition::NotAKnot) boundary in each level
///
/// ## Example
/// In a complex case all boundaries can be set individually:
/// ``` rust
/// # use ndarray_interp::*;
/// # use ndarray_interp::interp1d::*;
/// # use ndarray_interp::interp1d::cubic_spline::*;
/// # use ndarray::*;
/// # use approx::*;
///
/// let y = array![
///     [0.5, 1.0],
///     [0.0, 1.5],
///     [3.0, 0.5],
/// ];
/// let x = array![-1.0, 0.0, 3.0];
///
/// // first data column: natural
/// // second data column top: NotAKnot
/// // second data column bottom: first derivative == 0.5
/// let boundaries = array![
///     [
///         RowBoundary::Natural,
///         RowBoundary::Mixed { left: SingleBoundary::NotAKnot, right: SingleBoundary::FirstDeriv(0.5)}
///     ],
/// ];
/// let strat = CubicSpline::new().boundary(BoundaryCondition::Individual(boundaries));
/// let interpolator = Interp1DBuilder::new(y)
///     .x(x)
///     .strategy(strat)
///     .build().unwrap();
///
/// ```
#[derive(Debug, PartialEq, Eq)]
pub enum BoundaryCondition<T, D: Dimension> {
    /// Not a knot boundary. The first and second segment at a curve end are the same polynomial.
    NotAKnot,
    /// Natural boundary. The second derivative at the curve end is 0
    Natural,
    /// Clamped boundary. The first derivative at the curve end is 0
    Clamped,
    /// Periodic spline.
    /// The interpolated functions is assumed to be periodic.
    /// The first and last element in the data must be equal.
    Periodic,
    /// Set individual boundary conditions for each row in the data
    /// and/or individual conditions for the left and right boundary
    Individual(Array<RowBoundary<T>, D>),
}

impl<T, D: Dimension> Default for BoundaryCondition<T, D> {
    fn default() -> Self {
        Self::NotAKnot
    }
}

/// Boundary condition for a single data row
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum RowBoundary<T> {
    /// ![`BoundaryCondition::NotAKnot`]
    NotAKnot,
    /// ![`BoundaryCondition::Natural`]
    Natural,
    /// ![`BoundaryCondition::Clamped`]
    Clamped,
    /// ![`BoundaryCondition::Periodic`]
    Periodic,
    /// Set individual boundary conditions at the left and right end of the curve
    Mixed {
        left: SingleBoundary<T>,
        right: SingleBoundary<T>,
    },
}

impl<T: SplineNum> RowBoundary<T> {
    fn specialize(self) -> Self {
        use SingleBoundary::*;
        match self {
            RowBoundary::Natural => Self::Mixed {
                left: Natural,
                right: Natural,
            },
            RowBoundary::NotAKnot => Self::Mixed {
                left: NotAKnot,
                right: NotAKnot,
            },
            RowBoundary::Clamped => Self::Mixed {
                left: Clamped,
                right: Clamped,
            },
            _ => self,
        }
    }
}

impl<T: SplineNum> Default for RowBoundary<T> {
    fn default() -> Self {
        Self::NotAKnot
    }
}

/// Boundary condition for a single boundary (one side of one data row)
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SingleBoundary<T> {
    /// ![`BoundaryCondition::NotAKnot`]
    NotAKnot,
    /// This ist the same as `SingleBoundary::SecondDeriv(0.0)`
    /// ![`BoundaryCondition::Natural`]
    Natural,
    /// This ist the same as `SingleBoundary::FirstDeriv(0.0)`
    /// ![`BoundaryCondition::Clamped`]
    Clamped,
    /// Set a value for the first derivative at the curve end
    FirstDeriv(T),
    /// Set a value for the second derivative at the curve end
    SecondDeriv(T),
}

impl<T: SplineNum> SingleBoundary<T> {
    fn specialize(self) -> Self {
        use SingleBoundary::*;
        match self {
            SingleBoundary::Natural => SecondDeriv(cast(0.0).unwrap_or_else(|| unimplemented!())),
            SingleBoundary::Clamped => FirstDeriv(cast(0.0).unwrap_or_else(|| unimplemented!())),
            _ => self,
        }
    }
}

impl<T: SplineNum> Default for SingleBoundary<T> {
    fn default() -> Self {
        Self::NotAKnot
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
    /// Calculate the coefficients `a` and `b`
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
        let mut k = Array::zeros(dim.clone());
        let kv = k.view_mut();
        match self.boundary {
            BoundaryCondition::Periodic => todo!(),
            BoundaryCondition::Natural => Self::solve_for_k(kv, x, data, RowBoundary::Natural),
            BoundaryCondition::Clamped => Self::solve_for_k(kv, x, data, RowBoundary::Clamped),
            BoundaryCondition::NotAKnot => Self::solve_for_k(kv, x, data, RowBoundary::NotAKnot),
            BoundaryCondition::Individual(bounds) => {
                assert!(kv.raw_dim().remove_axis(AX0) == bounds.raw_dim().remove_axis(AX0)); // TODO: return error
                assert!(bounds.raw_dim()[0] == 1); // TODO: return error
                Self::solve_for_k_individual(
                    kv.into_dyn(),
                    x,
                    data.view().into_dyn(),
                    bounds.view().into_dyn(),
                );
            }
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

    fn solve_for_k_individual<Sx>(
        mut k: ArrayViewMut<T, IxDyn>,
        x: &ArrayBase<Sx, Ix1>,
        data: ArrayView<T, IxDyn>,
        boundary: ArrayView<RowBoundary<T>, IxDyn>,
    ) where
        Sx: Data<Elem = T>,
    {
        if k.ndim() > 1 {
            let ax = Axis(k.ndim() - 1);
            Zip::from(k.axis_iter_mut(ax))
                .and(data.axis_iter(ax))
                .and(boundary.axis_iter(ax))
                .for_each(|k, data, boundary| Self::solve_for_k_individual(k, x, data, boundary))
        } else {
            Self::solve_for_k(
                k,
                x,
                &data,
                boundary.first().cloned().unwrap_or_else(|| unreachable!()),
            )
        }
    }

    /// solves the linear equation `A * k = rhs` with the [`RowBoundary`] used for
    /// each row in the data
    ///  
    /// **returns** k
    fn solve_for_k<Sd, Sx, _D>(
        k: ArrayViewMut<T, _D>,
        x: &ArrayBase<Sx, Ix1>,
        data: &ArrayBase<Sd, _D>,
        boundary: RowBoundary<T>,
    ) where
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

        let zero: T = cast(0.0).unwrap_or_else(|| unimplemented!());
        let one: T = cast(1.0).unwrap_or_else(|| unimplemented!());
        let two: T = cast(2.0).unwrap_or_else(|| unimplemented!());
        let three: T = cast(3.0).unwrap_or_else(|| unimplemented!());

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
        match (boundary.specialize(), len) {
            (RowBoundary::Periodic, _) => todo!(),
            (RowBoundary::Clamped, _) => unreachable!(),
            (RowBoundary::Natural, _) => unreachable!(),
            (RowBoundary::NotAKnot, _) => unreachable!(),
            (
                RowBoundary::Mixed {
                    left: SingleBoundary::NotAKnot,
                    right: SingleBoundary::NotAKnot,
                },
                3,
            ) => {
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
            }
            (RowBoundary::Mixed { left, right }, _) => {
                match left.specialize() {
                    SingleBoundary::NotAKnot => {
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
                    }
                    SingleBoundary::Natural => unreachable!(),
                    SingleBoundary::Clamped => unreachable!(),
                    SingleBoundary::FirstDeriv(deriv) => {
                        a_mid[0] = one;
                        a_up[0] = zero;
                        rhs.index_axis_mut(AX0, 0).fill(deriv);
                    }
                    SingleBoundary::SecondDeriv(deriv) => {
                        let dx0 = x[1] - x[0];
                        a_up[0] = dx0;
                        a_mid[0] = two * dx0;
                        let rhs_0 = rhs.index_axis_mut(AX0, 0);
                        let data_0 = data.index_axis(AX0, 0);
                        let data_1 = data.index_axis(AX0, 1);
                        Zip::from(rhs_0)
                            .and(data_0)
                            .and(data_1)
                            .for_each(|rhs_0, &y_0, &y_1| {
                                *rhs_0 = three * (y_1 - y_0) - deriv * dx0.pow(two) / two;
                            });
                    }
                };
                match right.specialize() {
                    SingleBoundary::NotAKnot => {
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
                                *b = (dx_1.pow(two) * (y_2 - y_3) / dx_2
                                    + tmp1 * (y_1 - y_2) / dx_1)
                                    / d;
                            });
                    }
                    SingleBoundary::Natural => unreachable!(),
                    SingleBoundary::Clamped => unreachable!(),
                    SingleBoundary::FirstDeriv(deriv) => {
                        a_mid[len - 1] = one;
                        a_low[len - 1] = zero;
                        rhs.index_axis_mut(AX0, len - 1).fill(deriv);
                    }
                    SingleBoundary::SecondDeriv(deriv) => {
                        let dxn = x[len - 1] - x[len - 2];
                        a_mid[len - 1] = two * dxn;
                        a_low[len - 1] = dxn;
                        let rhs_n = rhs.index_axis_mut(AX0, len - 1);
                        let data_n = data.index_axis(AX0, len - 1);
                        let data_n1 = data.index_axis(AX0, len - 2);
                        Zip::from(rhs_n)
                            .and(data_n)
                            .and(data_n1)
                            .for_each(|rhs_n, &y_n, &y_n1| {
                                *rhs_n = three * (y_n - y_n1) + deriv * dxn.pow(two) / two;
                            });
                    }
                };
            }
        }
        Self::thomas(k, a_up, a_mid, a_low, rhs);
    }

    /// The Thomas algorithm is used, because the matrix A will be tridiagonal and diagonally dominant
    /// [https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm]
    fn thomas<_D>(
        mut k: ArrayViewMut<T, _D>,
        a_up: Array1<T>,
        mut a_mid: Array1<T>,
        a_low: Array1<T>,
        mut rhs: Array<T, _D>,
    ) where
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
