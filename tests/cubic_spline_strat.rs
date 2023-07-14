use approx::assert_abs_diff_eq;
use ndarray::{array, Array1};
use ndarray_interp::interp1d::{CubicSpline, Interp1D};
use ndarray_interp::{BuilderError, InterpolateError};

#[test]
fn interp() {
    let data = array![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0];
    let interp = Interp1D::builder(data)
        .strategy(CubicSpline)
        .build()
        .unwrap();
    let q = Array1::linspace(0.0, 11.0, 30);
    let res = interp.interp_array(&q).unwrap();

    // values from scipy.interpolate.QubicSpline
    let expect = array![
        1.0,
        1.3917082281418252,
        1.7709152572751259,
        2.125720997885402,
        2.4735200559559645,
        2.873596855334901,
        3.3692218872560726,
        3.822919531969092,
        3.998240261438613,
        3.75923077015136,
        3.279709933108678,
        2.7881342665115523,
        2.390891499049402,
        2.0569231634621636,
        1.744119027809967,
        1.38442936840091,
        0.8991930736934348,
        0.327385578986533,
        -0.01567970348252848,
        0.2056442153017282,
        0.9653909358084248,
        1.9164377865351583,
        2.757368677491977,
        3.485961877773172,
        4.197630493134489,
        4.947868508672761,
        5.71920917646552,
        6.487721497405632,
        7.246383891907155,
        8.0
    ];
    assert_abs_diff_eq!(res, expect, epsilon = f64::EPSILON);
}

#[test]
fn to_little_data() {
    let err = Interp1D::builder(array![1.0, 2.0])
        .strategy(CubicSpline)
        .build();
    assert!(matches!(err, Err(BuilderError::NotEnoughData(_))));
}

#[test]
fn enough_data() {
    Interp1D::builder(array![1.0, 2.0, 1.0])
        .strategy(CubicSpline)
        .build()
        .unwrap();
}

#[test]
fn extrapolate() {
    let interp = Interp1D::builder(array![1.0, 2.0, 1.0])
        .strategy(CubicSpline)
        .build()
        .unwrap();
    let err = interp.interp_scalar(-0.5);
    assert!(matches!(err, Err(InterpolateError::OutOfBounds(_))));
    let err = interp.interp_scalar(3.5);
    assert!(matches!(err, Err(InterpolateError::OutOfBounds(_))));
}
