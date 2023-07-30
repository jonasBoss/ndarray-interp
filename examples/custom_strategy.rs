use approx::assert_abs_diff_eq;
use ndarray::{array, Array, ArrayViewMut, Data, Dimension, RemoveAxis};
use ndarray_interp::{
    interp1d::{Interp1D, Interp1DStrategy, Interp1DStrategyBuilder},
    InterpolateError,
};

struct StepInterpolator;

impl<Sd, Sx, D> Interp1DStrategyBuilder<Sd, Sx, D> for StepInterpolator
where
    Sd: Data<Elem = f64>,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    const MINIMUM_DATA_LENGHT: usize = 2;

    type FinishedStrat = Self;

    fn build<Sx2>(
        self,
        _x: &ndarray::ArrayBase<Sx2, ndarray::Ix1>,
        _data: &ndarray::ArrayBase<Sd, D>,
    ) -> Result<Self::FinishedStrat, ndarray_interp::BuilderError>
    where
        Sx2: ndarray::Data<Elem = Sd::Elem>,
    {
        Ok(self)
    }
}

impl<Sd, Sx, D> Interp1DStrategy<Sd, Sx, D> for StepInterpolator
where
    Sd: Data<Elem = f64>,
    Sx: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
{
    fn interp_into(
        &self,
        interpolator: &Interp1D<Sd, Sx, D, Self>,
        mut target: ArrayViewMut<'_, Sd::Elem, D::Smaller>,
        x: Sx::Elem,
    ) -> Result<(), InterpolateError> {
        let idx = interpolator.get_index_left_of(x);
        let (x_left, data_left) = interpolator.index_point(idx);
        let (x_right, data_right) = interpolator.index_point(idx + 1);
        if (x_right - x_left) / 2.0 > (x - x_left) {
            target.assign(&data_left);
        } else {
            target.assign(&data_right);
        }
        Ok(())
    }
}

fn main() {
    let data = array![2.0, 4.0, 5.0];
    let query = Array::linspace(-0.5, 2.5, 6);

    let interp = Interp1D::builder(data)
        .strategy(StepInterpolator)
        .build()
        .unwrap();

    let result = interp.interp_array(&query).unwrap();
    let expect = array![2.0, 2.0, 4.0, 4.0, 5.0, 5.0];
    assert_abs_diff_eq!(result, expect, epsilon = f64::EPSILON);
}
