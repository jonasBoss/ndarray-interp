// this includes the test for the linear strategy as well.
// Because the linear strategy is used to test a lot of
// different behaviour for `Interp1D`

use approx::assert_abs_diff_eq;
use ndarray::array;
use ndarray::s;
use num_traits::NumCast;

use ndarray_interp::interp1d::{Interp1D, Interp1DBuilder, Linear};
use ndarray_interp::BuilderError;
use ndarray_interp::InterpolateError;

#[test]
fn test_type_cast_assumptions() {
    assert_eq!(<i32 as NumCast>::from(1.75).unwrap(), 1);
    assert_eq!(<i32 as NumCast>::from(1.25).unwrap(), 1);
}

#[test]
fn interp_y_only() {
    let interp = Interp1D::builder(array![1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.5])
        .build()
        .unwrap();
    assert_eq!(*interp.interp(0.0).unwrap().first().unwrap(), 1.5);
    assert_eq!(*interp.interp(9.0).unwrap().first().unwrap(), 10.5);
    assert_eq!(*interp.interp(4.5).unwrap().first().unwrap(), 6.0);
    assert_eq!(*interp.interp(0.25).unwrap().first().unwrap(), 1.625);
    assert_eq!(*interp.interp(8.75).unwrap().first().unwrap(), 10.125);
}

#[test]
fn extrapolate_y_only() {
    let interp = Interp1D::builder(array![1.0, 2.0, 1.5])
        .strategy(Linear::new().extrapolate(true))
        .build()
        .unwrap();
    assert_eq!(*interp.interp(-1.0).unwrap().first().unwrap(), 0.0);
    assert_eq!(*interp.interp(3.0).unwrap().first().unwrap(), 1.0);
}

#[test]
fn interp_with_x_and_y() {
    let interp = Interp1DBuilder::new(array![1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.5])
        .x(array![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .strategy(Linear::new())
        .build()
        .unwrap();
    assert_eq!(*interp.interp(-4.0).unwrap().first().unwrap(), 1.5);
    assert_eq!(*interp.interp(5.0).unwrap().first().unwrap(), 10.5);
    assert_eq!(*interp.interp(0.5).unwrap().first().unwrap(), 6.0);
    assert_eq!(*interp.interp(-3.75).unwrap().first().unwrap(), 1.625);
    assert_eq!(*interp.interp(4.75).unwrap().first().unwrap(), 10.125);
}

#[test]
fn interp_with_x_and_y_expspaced() {
    let interp = Interp1DBuilder::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        .x(array![
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0
        ])
        .strategy(Linear::new())
        .build()
        .unwrap();
    assert_eq!(*interp.interp(1.0).unwrap().first().unwrap(), 1.0);
    assert_eq!(*interp.interp(512.0).unwrap().first().unwrap(), 1.0);
    assert_eq!(*interp.interp(42.0).unwrap().first().unwrap(), 4.6875);
    assert_eq!(*interp.interp(365.0).unwrap().first().unwrap(), 1.57421875);
}

#[test]
fn extrapolate_with_x_and_y() {
    let interp = Interp1DBuilder::new(array![1.0, 0.0, 1.5])
        .x(array![0.0, 1.0, 1.5])
        .strategy(Linear::new().extrapolate(true))
        .build()
        .unwrap();
    assert_eq!(*interp.interp(-1.0).unwrap().first().unwrap(), 2.0);
    assert_eq!(*interp.interp(2.0).unwrap().first().unwrap(), 3.0);
}

#[test]
fn interp_array() {
    let interp = Interp1D::builder(array![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        .build()
        .unwrap();
    let x_query = array![[1.0, 2.0, 9.0], [4.0, 5.0, 7.5]];
    let y_expect = array![[2.0, 3.0, 1.0], [5.0, 5.0, 2.5]];
    assert_eq!(interp.interp_array(&x_query).unwrap(), y_expect);
}

#[test]
fn interp_y_only_out_of_bounds() {
    let interp = Interp1D::builder(array![1.0, 2.0, 3.0]).build().unwrap();
    assert!(matches!(
        interp.interp(-0.1),
        Err(InterpolateError::OutOfBounds(_))
    ));
    assert!(matches!(
        interp.interp(9.0),
        Err(InterpolateError::OutOfBounds(_))
    ));
}

#[test]
fn interp_with_x_and_y_out_of_bounds() {
    let interp = Interp1DBuilder::new(array![1.0, 2.0, 3.0])
        .x(array![-4.0, -3.0, 2.0])
        .strategy(Linear::new())
        .build()
        .unwrap();
    assert!(matches!(
        interp.interp(-4.1),
        Err(InterpolateError::OutOfBounds(_))
    ));
    assert!(matches!(
        interp.interp(2.1),
        Err(InterpolateError::OutOfBounds(_))
    ));
}

#[test]
fn interp_builder_errors() {
    assert!(matches!(
        Interp1DBuilder::new(array![1]).build(),
        Err(BuilderError::NotEnoughData(_))
    ));
    assert!(matches!(
        Interp1DBuilder::new(array![1, 2])
            .x(array![1, 2, 3])
            .build(),
        Err(BuilderError::AxisLenght(_))
    ));
    assert!(matches!(
        Interp1DBuilder::new(array![1, 2, 3])
            .x(array![1, 2, 2])
            .build(),
        Err(BuilderError::Monotonic(_))
    ));
}

#[test]
fn interp_view_array() {
    let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let interp = Interp1D::builder(a.slice(s![..;-1]))
        .x(array![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .build()
        .unwrap();
    println!("{:?}", interp.interp(5.0).unwrap());
    assert_eq!(*interp.interp(-4.0).unwrap().first().unwrap(), 10.0);
    assert_eq!(*interp.interp(5.0).unwrap().first().unwrap(), 1.0);
    assert_eq!(*interp.interp(0.0).unwrap().first().unwrap(), 6.0);
    assert_eq!(*interp.interp(-3.5).unwrap().first().unwrap(), 9.5);
    assert_eq!(*interp.interp(4.75).unwrap().first().unwrap(), 1.25);
}

#[test]
fn interp_multi_fn() {
    let data = array![
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [2.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [20.0, 40.0, 60.0, 80.0, 100.0],
    ];
    let interp = Interp1DBuilder::new(data)
        .x(array![1.0, 2.0, 3.0, 4.0])
        .build()
        .unwrap();
    let res = interp.interp(1.5).unwrap();
    assert_abs_diff_eq!(
        res,
        array![1.05, 1.1, 1.65, 2.2, 2.75],
        epsilon = f64::EPSILON
    );
    let array_array = interp
        .interp_array(&array![[1.0, 1.5], [3.5, 4.0]])
        .unwrap();

    assert_abs_diff_eq!(
        array_array.slice(s![1, 1, ..]),
        array![20.0, 40.0, 60.0, 80.0, 100.0],
        epsilon = f64::EPSILON
    );
    assert_abs_diff_eq!(
        array_array,
        array![
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.05, 1.1, 1.65, 2.2, 2.75]],
            [
                [15.0, 30.0, 45.0, 60.0, 75.0],
                [20.0, 40.0, 60.0, 80.0, 100.0]
            ]
        ],
        epsilon = f64::EPSILON
    );
}

#[test]
fn interp_array_with_differnt_repr() {
    let interp = Interp1D::builder(array![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        .build()
        .unwrap();
    let x_query = array![[1.0, 2.0, 9.0], [4.0, 5.0, 7.5]];
    let y_expect = array![[2.0, 3.0, 1.0], [5.0, 5.0, 2.5]];
    assert_eq!(interp.interp_array(&x_query.view()).unwrap(), y_expect);
}
