// this includes the test for the bilinear strategy as well.
// Because the bilinear strategy is used to test a lot of
// different behaviour for `Interp2D`


use std::iter::repeat;

use approx::assert_abs_diff_eq;
use ndarray::{array, Array2, Array};
use ndarray_interp::{interp2d::{Interp2D, Interp2DBuilder}, BuilderError, InterpolateError};

fn data_i32() -> Array2<i32>{
    array![
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
    ]
}

fn data_f64() -> Array2<f64>{
    array![
        [1.0,2.0,3.0,4.0],
        [5.0,6.0,7.0,8.0],
        [9.0,10.0,11.0,12.0],
    ]
}

#[test]
fn cornerns_only_data_no_axis() {
    let data =data_i32();
    let interp = Interp2D::builder(data).build().unwrap();
    assert_eq!(interp.interp_scalar(0, 0).unwrap(), 1);
    assert_eq!(interp.interp_scalar(2, 3).unwrap(), 12);
    assert_eq!(interp.interp_scalar(2, 0).unwrap(), 9);
    assert_eq!(interp.interp_scalar(0, 3).unwrap(), 4);
}

#[test]
fn cornerns_only_x_axis() {
    let data = data_i32();
    let interp = Interp2D::builder(data.view())
        .x(array![1,2,3])
        .build()
        .unwrap();
    assert_eq!(interp.interp_scalar(1, 0).unwrap(), 1);
    assert_eq!(interp.interp_scalar(3, 3).unwrap(), 12);
    assert_eq!(interp.interp_scalar(3, 0).unwrap(), 9);
    assert_eq!(interp.interp_scalar(1, 3).unwrap(), 4);
}

#[test]
fn cornerns_only_y_axis() {
    let data = data_f64();
    let interp = Interp2D::builder(data.view())
        .y(array![-3.0,-2.0,-1.0,0.0])
        .build()
        .unwrap();
    assert_eq!(interp.interp_scalar(0.0, -3.0).unwrap(), 1.0);
    assert_eq!(interp.interp_scalar(2.0, 0.0).unwrap(), 12.0);
    assert_eq!(interp.interp_scalar(2.0, -3.0).unwrap(), 9.0);
    assert_eq!(interp.interp_scalar(0.0, 0.0).unwrap(), 4.0);
}

#[test]
fn extrapolate() {
    let data =data_i32();
    let interp = Interp2D::builder(data).build().unwrap();
    assert!(matches!(
        interp.interp_scalar(-1, 1),
        Err(InterpolateError::OutOfBounds(_))
    ));
    assert!(matches!(
        interp.interp_scalar(1, -1),
        Err(InterpolateError::OutOfBounds(_))
    ));
    assert!(matches!(
        interp.interp_scalar(3, 1),
        Err(InterpolateError::OutOfBounds(_))
    ));
    assert!(matches!(
        interp.interp_scalar(1, 4),
        Err(InterpolateError::OutOfBounds(_))
    ));
}

#[test]
fn interpolate_array(){
    let data = Array::linspace(0.0, 8.0, 9).into_shape((3,3)).unwrap();
    let x = array![1.0,2.0,3.0];
    let y = array![4.0,5.0,6.0];
    let resolution = 11usize;
    let qx = Array::linspace(1.0, 3.0, resolution);
    let qy = Array::linspace(4.0, 6.0, resolution);
    let qx = Array::from_iter(qx.into_iter().flat_map(|x| repeat(x).take(resolution))).into_shape((resolution,resolution)).unwrap();
    let qy = Array::from_iter(repeat(qy).take(resolution).flatten()).into_shape((resolution,resolution)).unwrap();

    let interp = Interp2D::builder(data)
        .x(x)
        .y(y)
        .build()
        .unwrap();

    let res = interp.interp_array(&qx, &qy).unwrap();
    let expect = array![
        [0.0, 0.20000000000000018, 0.40000000000000036, 0.5999999999999996, 0.7999999999999998, 1.0, 1.2000000000000002, 1.4000000000000004, 1.5999999999999996, 1.7999999999999998, 2.0],
        [0.5999999999999999, 0.8, 1.0000000000000002, 1.1999999999999995, 1.3999999999999997, 1.5999999999999999, 1.8, 2.0, 2.1999999999999993, 2.3999999999999995, 2.5999999999999996],
        [1.1999999999999997, 1.4, 1.6, 1.7999999999999994, 1.9999999999999996, 2.1999999999999997, 2.4, 2.6, 2.7999999999999994, 2.9999999999999996, 3.1999999999999997],
        [1.8000000000000003, 2.0000000000000004, 2.2000000000000006, 2.4, 2.6, 2.8000000000000003, 3.0000000000000004, 3.2000000000000006, 3.4, 3.6, 3.8000000000000003],
        [2.4000000000000004, 2.6000000000000005, 2.8000000000000007, 3.0, 3.2, 3.4000000000000004, 3.6000000000000005, 3.8000000000000007, 4.0, 4.2, 4.4],
        [3.0, 3.2, 3.4000000000000004, 3.5999999999999996, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0],
        [3.6000000000000005, 3.8000000000000007, 4.000000000000001, 4.2, 4.4, 4.6000000000000005, 4.800000000000001, 5.000000000000001, 5.2, 5.4, 5.6000000000000005],
        [4.200000000000001, 4.400000000000001, 4.600000000000001, 4.800000000000001, 5.000000000000001, 5.200000000000001, 5.400000000000001, 5.600000000000001, 5.800000000000001, 6.000000000000001, 6.200000000000001],
        [4.800000000000001, 5.000000000000001, 5.200000000000001, 5.4, 5.6000000000000005, 5.800000000000001, 6.000000000000001, 6.200000000000001, 6.4, 6.6000000000000005, 6.800000000000001],
        [5.3999999999999995, 5.6, 5.8, 5.999999999999999, 6.199999999999999, 6.3999999999999995, 6.6, 6.8, 6.999999999999999, 7.199999999999999, 7.3999999999999995],
        [6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
    ];
    assert_abs_diff_eq!(res, expect, epsilon=f64::EPSILON)
}


#[test]
fn interp_nd_data(){
    let data = Array::from_iter(     
        [[[[1.0,10.0],[-1.0,-10.0]], [[2.0,20.0],[-2.0,-20.0]]],
        [[[3.0,30.0],[-3.0,-30.0]], [[5.0,50.0],[-5.0,-50.0]]]].into_iter().flatten().flatten().flatten()
    ).into_shape((2,2,2,2)).unwrap();

    let interp = Interp2DBuilder::new(data).build().unwrap();
    let res = interp.interp(0.0,0.5).unwrap();
    let expect = array![[1.5,15.0],[-1.5,-15.0]];
    assert_abs_diff_eq!(res, expect, epsilon=f64::EPSILON);

    let qx = array![0.0,0.5];
    let qy = array![0.5,1.0];
    let expect = array![
        [[1.5,15.0], [-1.5,-15.0]],
        [[3.5,35.0], [-3.5,-35.0]]
    ];
    let res = interp.interp_array(&qx, &qy).unwrap();
    assert_abs_diff_eq!(res, expect, epsilon=f64::EPSILON);
}

#[test]
#[should_panic(expected = "`xs.shape()` and `ys.shape()` do not match")]
fn interp_array_with_unmatched_axis(){
    let data = Array::linspace(0.0, 8.0, 9).into_shape((3,3)).unwrap();
    let qx = array![0.0,1.0];
    let qy =array![0.0,1.0,2.0];
    let interp = Interp2D::builder(data).build().unwrap();
    let _res = interp.interp_array(&qx, &qy);
}

#[test]
fn builder_errors(){
    assert!(matches!(
        Interp2D::builder(array![[1]]).build(),
        Err(BuilderError::NotEnoughData(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1,2]]).build(),
        Err(BuilderError::NotEnoughData(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1],[2]]).build(),
        Err(BuilderError::NotEnoughData(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1,2],[3,4]])
        .x(array![1])
        .build(),
        Err(BuilderError::AxisLenght(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1,2],[3,4]])
        .x(array![1,2,3])
        .build(),
        Err(BuilderError::AxisLenght(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1,2],[3,4]])
        .y(array![1])
        .build(),
        Err(BuilderError::AxisLenght(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1,2],[3,4]])
        .y(array![1,2,3])
        .build(),
        Err(BuilderError::AxisLenght(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1,2],[3,4]])
        .x(array![2,2])
        .build(),
        Err(BuilderError::Monotonic(_))
    ));
    assert!(matches!(
        Interp2D::builder(array![[1,2],[3,4]])
        .y(array![2,2])
        .build(),
        Err(BuilderError::Monotonic(_))
    ));

}
