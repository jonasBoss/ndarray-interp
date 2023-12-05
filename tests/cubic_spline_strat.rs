use approx::assert_relative_eq;
use ndarray::{array, Array1};
use ndarray_interp::interp1d::{
    BoundaryCondition, CubicSpline, Interp1D, Interp1DBuilder, RowBoundary, SingleBoundary,
};
use ndarray_interp::{BuilderError, InterpolateError};

#[test]
fn interp_natural() {
    let data = array![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0];
    let interp = Interp1D::builder(data)
        .strategy(CubicSpline::new().boundary(BoundaryCondition::Natural))
        .build()
        .unwrap();
    let q = Array1::linspace(0.0, 11.0, 30);
    let res = interp.interp_array(&q).unwrap();

    // values from scipy.interpolate.QubicSpline with bc_type="natural"
    let expect = array![
        1., 1.39170823, 1.77091526, 2.125721, 2.47352006, 2.87359686, 3.36922189, 3.82291953,
        3.99824026, 3.75923077, 3.27970993, 2.78813427, 2.3908915, 2.05692316, 1.74411903,
        1.38442937, 0.89919307, 0.32738558, -0.0156797, 0.20564422, 0.96539094, 1.91643779,
        2.75736868, 3.48596188, 4.19763049, 4.94786851, 5.71920918, 6.4877215, 7.24638389, 8.
    ];
    assert_relative_eq!(res, expect, epsilon = f64::EPSILON, max_relative = 0.001);
}

#[test]
fn to_little_data() {
    let err = Interp1D::builder(array![1.0, 2.0])
        .strategy(CubicSpline::new())
        .build();
    assert!(matches!(err, Err(BuilderError::NotEnoughData(_))));
}

#[test]
fn enough_data() {
    Interp1D::builder(array![1.0, 2.0, 1.0])
        .strategy(CubicSpline::new())
        .build()
        .unwrap();
}

#[test]
fn extrapolate_false() {
    let interp = Interp1D::builder(array![1.0, 2.0, 1.0])
        .strategy(CubicSpline::new())
        .build()
        .unwrap();
    let err = interp.interp(-0.5);
    assert!(matches!(err, Err(InterpolateError::OutOfBounds(_))));
    let err = interp.interp(3.5);
    assert!(matches!(err, Err(InterpolateError::OutOfBounds(_))));
}

#[test]
fn extrapolate_natural() {
    let data = array![1.0, 2.0, 2.5, 2.5, 3.0, 2.0, 1.0, -2.0, 3.0, 5.0, 6.3, 8.0];
    let interp = Interp1D::builder(data)
        .strategy(
            CubicSpline::new()
                .extrapolate(true)
                .boundary(BoundaryCondition::Natural),
        )
        .build()
        .unwrap();
    let q = Array1::linspace(-3.0, 15.0, 30);
    let res = interp.interp_array(&q).unwrap();

    // values from scipy.interpolate.QubicSpline with bc_type="natural"
    let expect = array![
        -0.10117811,
        -0.50187696,
        -0.46744049,
        -0.11138225,
        0.45278419,
        1.11154527,
        1.75138741,
        2.25775994,
        2.49749363,
        2.442418,
        2.62405156,
        3.00988064,
        2.60389947,
        1.96187505,
        1.6459892,
        -0.21920517,
        -2.0380548,
        0.35839389,
        3.69754559,
        4.82435282,
        5.45047974,
        6.35498498,
        7.39691304,
        8.48312564,
        9.5339106,
        10.46955574,
        11.21034887,
        11.67657779,
        11.78853034,
        11.46649431
    ];
    assert_relative_eq!(res, expect, epsilon = f64::EPSILON, max_relative = 0.001);
}

#[test]
fn extrapolate_not_a_knot() {
    let data = array![1f32, 2.0, 2.5, 2.5, 3.0, 2.0, 1.0, -2.0, 3.0, 5.0, 6.3, 8.0];
    let interp = Interp1D::builder(data)
        .strategy(
            CubicSpline::new()
                .extrapolate(true)
                .boundary(BoundaryCondition::NotAKnot),
        )
        .build()
        .unwrap();
    let q = Array1::linspace(-3.0, 15.0, 30);
    let res = interp.interp_array(&q).unwrap();
    // values from scipy.interpolate.QubicSpline with bc_type="not-a-knot"
    let expect = array![
        0.94398816f32,
        0.09886458,
        -0.16503997,
        0.01013939,
        0.48226752,
        1.109_209_3,
        1.748_829_5,
        2.258_993_1,
        2.497_564_8,
        2.4421474,
        2.624_124,
        3.009_909_2,
        2.603_880_2,
        1.961_875_3,
        1.645_976_8,
        -0.21916762,
        -2.038_032_5,
        0.35816476,
        3.697_835_4,
        4.825_070_4,
        5.447_815_4,
        6.3556859,
        7.409_040_5,
        8.452_749,
        9.331_685,
        9.890_717,
        9.974_716,
        9.428_555,
        8.097_102,
        5.825_231
    ];
    assert_relative_eq!(res, expect, epsilon = f32::EPSILON, max_relative = 0.001);
}

#[test]
fn not_a_knot_3_values() {
    let interp = Interp1D::builder(array![1.0, 2.0, 0.0])
        .strategy(
            CubicSpline::new()
                .boundary(BoundaryCondition::NotAKnot)
                .extrapolate(true),
        )
        .build()
        .unwrap();

    let q = Array1::linspace(-1.0, 3.0, 15);
    let res = interp.interp_array(&q).unwrap();

    let expect = array![
        -3.,
        -1.55102041,
        -0.34693878,
        0.6122449,
        1.32653061,
        1.79591837,
        2.02040816,
        2.,
        1.73469388,
        1.2244898,
        0.46938776,
        -0.53061224,
        -1.7755102,
        -3.26530612,
        -5.
    ];
    assert_relative_eq!(res, expect, epsilon = f64::EPSILON, max_relative = 0.001);
}

#[test]
fn multidim_multi_bounds() {
    let y = array![[0.5, 1.0], [0.0, 1.5], [3.0, 0.5],];
    let x = array![-1.0, 0.0, 3.0];

    // first data column: natural
    // second data column top: NotAKnot
    // second data column bottom: first derivative == 0.5
    let boundaries = array![[
        RowBoundary::Natural,
        RowBoundary::Mixed {
            left: SingleBoundary::NotAKnot,
            right: SingleBoundary::FirstDeriv(0.5)
        }
    ],];
    let strat = CubicSpline::new().boundary(BoundaryCondition::Individual(boundaries));
    let interpolator = Interp1DBuilder::new(y)
        .x(x)
        .strategy(strat)
        .build()
        .unwrap();
}
