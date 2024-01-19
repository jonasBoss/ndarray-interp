use approx::assert_relative_eq;
use ndarray::{array, stack, Array1, Axis};
use ndarray_interp::interp1d::cubic_spline::{
    BoundaryCondition, CubicSpline, RowBoundary, SingleBoundary,
};
use ndarray_interp::interp1d::{Interp1D, Interp1DBuilder};
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
    let strat = CubicSpline::new()
        .boundary(BoundaryCondition::Individual(boundaries))
        .extrapolate(true);
    let interp = Interp1DBuilder::new(y)
        .x(x)
        .strategy(strat)
        .build()
        .unwrap();

    let query = Array1::linspace(-2.0, 4.0, 15);
    let res = interp.interp_array(&query).unwrap();

    let expect = stack![
        Axis(1),
        [
            1.,
            0.85787172,
            0.59766764,
            0.30794461,
            0.07725948,
            -0.00655977,
            0.10058309,
            0.375,
            0.78717201,
            1.30758017,
            1.90670554,
            2.55502915,
            3.22303207,
            3.88119534,
            4.5
        ],
        [
            -1.13194444,
            0.02834467,
            0.81235828,
            1.27749433,
            1.48115079,
            1.48072562,
            1.33361678,
            1.09722222,
            0.82893991,
            0.5861678,
            0.42630385,
            0.40674603,
            0.58489229,
            1.01814059,
            1.76388889,
        ]
    ];
    assert_relative_eq!(res, expect, epsilon = f64::EPSILON, max_relative = 0.001);
}

#[test]
fn extrapolate_clamped() {
    let data = array![1.0, 2.0, 2.5, 2.5, 3.0, 2.0, 1.0, -2.0, 3.0, 5.0, 6.3, 8.0];
    let interp = Interp1D::builder(data)
        .strategy(
            CubicSpline::new()
                .extrapolate(true)
                .boundary(BoundaryCondition::Clamped),
        )
        .build()
        .unwrap();
    let q = Array1::linspace(-3.0, 15.0, 30);
    let res = interp.interp_array(&q).unwrap();

    // values from scipy.interpolate.QubicSpline with bc_type="clamped"
    let expect = array![
        41.28722497,
        23.28738691,
        11.50757146,
        4.70085655,
        1.6203201,
        1.01904002,
        1.65009422,
        2.30659337,
        2.50031574,
        2.43169729,
        2.62693014,
        3.01102652,
        2.60307096,
        1.96191635,
        1.64574608,
        -0.21831221,
        -2.03751124,
        0.35279783,
        3.70463099,
        4.84190082,
        5.38534268,
        6.37212173,
        7.69341241,
        7.7404559,
        4.5896631,
        -3.68255511,
        -18.99978784,
        -43.28562421,
        -78.46365334,
        -126.45746433
    ];
    assert_relative_eq!(res, expect, epsilon = f64::EPSILON, max_relative = 0.001);
}

#[test]
fn extrapolate_deriv1() {
    let data = array![1.0, 2.0, 2.5, 2.5, 3.0, 2.0, 1.0, -2.0, 3.0, 5.0, 6.3, 8.0];
    let interp = Interp1D::builder(data)
        .strategy(
            CubicSpline::new()
                .extrapolate(true)
                .boundary(BoundaryCondition::Individual(array![RowBoundary::Mixed {
                    left: SingleBoundary::FirstDeriv(-0.1),
                    right: SingleBoundary::FirstDeriv(-0.5)
                },])),
        )
        .build()
        .unwrap();
    let q = Array1::linspace(-3.0, 15.0, 30);
    let res = interp.interp_array(&q).unwrap();

    // values from scipy.interpolate.QubicSpline with bc_type=((1,-0.1),(1,-0.5))
    let expect = array![
        45.12263976,
        25.49190916,
        12.61728065,
        5.14680023,
        1.72851392,
        1.01046772,
        1.64070764,
        2.31111841,
        2.50057718,
        2.43070534,
        2.62719459,
        3.01112854,
        2.60301259,
        1.96191065,
        1.64564649,
        -0.2180452,
        -2.03735486,
        0.35120098,
        3.70664967,
        4.84689904,
        5.36679077,
        6.37700245,
        7.77785832,
        7.52893643,
        3.18149421,
        -7.71321086,
        -27.60392136,
        -58.93937981,
        -104.16832878,
        -165.7395108
    ];
    assert_relative_eq!(res, expect, epsilon = f64::EPSILON, max_relative = 0.001);
}

#[test]
fn extrapolate_deriv2() {
    let data = array![1.0, 2.0, 2.5, 2.5, 3.0, 2.0, 1.0, -2.0, 3.0, 5.0, 6.3, 8.0];
    let interp = Interp1D::builder(data)
        .strategy(
            CubicSpline::new()
                .extrapolate(true)
                .boundary(BoundaryCondition::Individual(array![RowBoundary::Mixed {
                    left: SingleBoundary::SecondDeriv(-0.1),
                    right: SingleBoundary::SecondDeriv(-0.5)
                },])),
        )
        .build()
        .unwrap();
    let q = Array1::linspace(-3.0, 15.0, 30);
    let res = interp.interp_array(&q).unwrap();

    // values from scipy.interpolate.QubicSpline with bc_type=((2,-0.1),(2,-0.5))
    let expect = array![
        -1.20835424,
        -1.1382612,
        -0.78778322,
        -0.24011435,
        0.42155137,
        1.11401989,
        1.75409718,
        2.25645344,
        2.49741809,
        2.44270565,
        2.62397325,
        3.00984762,
        2.60393207,
        1.96186855,
        1.645952,
        -0.21912456,
        -2.03800922,
        0.35793208,
        3.69812853,
        4.82579579,
        5.4451242,
        6.35639393,
        7.42129049,
        8.42206522,
        9.12740733,
        9.306006,
        8.72655042,
        7.15772979,
        4.36823329,
        0.12675012
    ];
    assert_relative_eq!(res, expect, epsilon = f64::EPSILON, max_relative = 0.001);
}

#[test]
#[should_panic(expected = "Expected: [1, 2], got: [1, 3]")]
fn bounds_shape_error1() {
    let y = array![[0.5, 1.0], [0.0, 1.5], [3.0, 0.5],];
    let boundaries = BoundaryCondition::Individual(array![[
        RowBoundary::Natural,
        RowBoundary::Clamped,
        RowBoundary::NotAKnot
    ],]);
    Interp1DBuilder::new(y)
        .strategy(CubicSpline::new().boundary(boundaries))
        .build()
        .unwrap();
}

#[test]
#[should_panic(expected = "Expected: [1, 2], got: [2, 2]")]
fn bounds_shape_error2() {
    let y = array![[0.5, 1.0], [0.0, 1.5], [3.0, 0.5],];
    let boundaries = BoundaryCondition::Individual(array![
        [RowBoundary::Natural, RowBoundary::NotAKnot],
        [RowBoundary::Natural, RowBoundary::NotAKnot],
    ]);
    Interp1DBuilder::new(y)
        .strategy(CubicSpline::new().boundary(boundaries))
        .build()
        .unwrap();
}

#[test]
#[should_panic(
    expected = "First: [0.5, 1.0], shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1, last: [0.5, 1.1]"
)]
fn periodic_wrong_values() {
    let y = array![[0.5, 1.0], [0.0, 1.5], [0.5, 1.1],];
    Interp1DBuilder::new(y)
        .strategy(CubicSpline::new().boundary(BoundaryCondition::Periodic))
        .build()
        .unwrap();
}

#[test]
fn extrapolate_periodic() {
    let data = array![1.0, 2.0, 2.5, 2.5, 3.0, 2.0, 1.0, -2.0, 3.0, 5.0, 6.3, 1.0];
    let interp = Interp1D::builder(data)
        .strategy(
            CubicSpline::new()
                .extrapolate(true)
                .boundary(BoundaryCondition::Periodic),
        )
        .build()
        .unwrap();
}
