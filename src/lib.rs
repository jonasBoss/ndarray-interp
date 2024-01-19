// Copyright (c) 2023 Jonas Bosse
//
// Licensed under the MIT license

//! The ndarray-interp crate provides interpolation algorithms
//! for interpolating _n_-dimesional data.
//!
//! # 1D Interpolation
//! The [interp1d] module provides the [`Interp1D`](interp1d::Interp1D) interpolator
//! and different interpolation strategies
//!
//! **1D Strategies**
//!  - [`interp1d::Linear`] - Linear interpolation and extrapolation
//!  - [`interp1d::cubic_spline`] - Cubic Spline interpolation with different boundary conditions.
//!
//! # 2D Interpolation
//! The [interp2d] module provides the [`Interp2D`](interp2d::Interp2D) interpolator
//! and different interpolation strategies
//!
//! **2D Strategies**
//!  - [`interp2d::Bilinear`] - Bilinear interpolation and extrapolation
//!
//! # Custom interpolation strategy
//! This crate defines traits to allow implementation of user
//! defined interpolation algorithms.
//! A 1D interpolation strategy can be created by implementing the
//! [`Interp1DStrategy`](interp1d::Interp1DStrategy) and
//! [`Interp1DStrategyBuilder`](interp1d::Interp1DStrategyBuilder) traits.
//! A 2D interpolation strategy can be created by implementing the
//! [`Interp2DStrategy`](interp2d::Interp2DStrategy) and
//! [`Interp2DStrategyBuilder`](interp2d::Interp2DStrategyBuilder) traits.
//!
//! See also the `custom_strategy.rs` example.
//!
//! # Examples
//! **1D Example**
//! ``` rust
//! use ndarray_interp::interp1d::*;
//! use ndarray::*;
//!
//! let data = array![0.0, 1.0, 1.5, 1.0, 0.0 ];
//! let interp = Interp1DBuilder::new(data).build().unwrap();
//!
//! let result = interp.interp_scalar(3.5).unwrap();
//! assert!(result == 0.5);
//! let result = interp.interp_array(&array![0.0, 0.5, 1.5]).unwrap();
//! assert!(result == array![0.0, 0.5, 1.25])
//! ```
//!
//! **1D Example with multidimensional data**
//! ```rust
//! use ndarray_interp::interp1d::*;
//! use ndarray::*;
//!
//! let data = array![
//!     [0.0, 1.0],
//!     [1.0, 2.0],
//!     [1.5, 2.5],
//!     [1.0, 2.0],
//! ];
//! let x = array![1.0, 2.0, 3.0, 4.0];
//!
//! let interp = Interp1D::builder(data)
//!     .strategy(Linear::new().extrapolate(true))
//!     .x(x)
//!     .build().unwrap();
//!
//! let result = interp.interp(0.5).unwrap();
//! assert!(result == array![-0.5, 0.5]);
//! let result = interp.interp_array(&array![0.5, 4.0]).unwrap();
//! assert!(result == array![[-0.5, 0.5], [1.0, 2.0]]);
//! ```
//!
//! **2D Example**
//! ```rust
//! use ndarray_interp::interp2d::*;
//! use ndarray::*;
//!
//! let data = array![
//!     [1.0, 2.0, 2.5],
//!     [3.0, 4.0, 3.5],
//! ];
//! let interp = Interp2D::builder(data).build().unwrap();
//!
//! let result = interp.interp_scalar(0.0, 0.5).unwrap();
//! assert!(result == 1.5);
//! let result = interp.interp_array(&array![0.0, 1.0], &array![0.5, 2.0]).unwrap();
//! assert!(result == array![1.5, 3.5]);
//! ```
//!
//! **1D Example with multidimensional data**
//! ``` rust
//! use ndarray_interp::interp2d::*;
//! use ndarray::*;
//!
//! let data = array![
//!     // ---------------------------------> y
//!     [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], // |
//!     [[4.0, -4.0], [5.0, -5.0], [6.0, -6.0]], // |
//!     [[7.0, -7.0], [8.0, -8.0], [9.0, -9.0]], // V
//!     [[7.5, -7.5], [8.5, -8.5], [9.5, -9.5]], // x
//! ];
//! let x = array![1.0, 2.0, 3.0, 4.0];
//! let y = array![1.0, 2.0, 3.0];
//!
//! let interp = Interp2D::builder(data)
//!     .x(x)
//!     .y(y)
//!     .build().unwrap();
//!
//! let result = interp.interp(1.5, 2.0).unwrap();
//! assert!(result == array![3.5, -3.5]);
//! let result = interp.interp_array(&array![1.5, 1.5], &array![2.0, 2.5]).unwrap();
//! assert!(result == array![[3.5, -3.5],[4.0, -4.0]]);
//! ```

use std::mem::ManuallyDrop;

use thiserror::Error;

mod dim_extensions;
pub mod interp1d;
pub mod interp2d;
pub mod vector_extensions;

/// Errors during Interpolator creation
#[derive(Debug, Error)]
pub enum BuilderError {
    /// Insufficient data for the chosen interpolation strategy
    #[error("{0}")]
    NotEnoughData(String),
    /// A interpolation axis is not strict monotonic rising
    #[error("{0}")]
    Monotonic(String),
    #[error("{0}")]
    ShapeError(String),
    #[error("{0}")]
    ValueError(String),
}

/// Errors during Interpolation
#[derive(Debug, Error)]
pub enum InterpolateError {
    #[error("{0}")]
    OutOfBounds(String),
}

/// cast `a` from type `A` to type `B` without any safety checks
///
/// ## Safety
///  - The caller must guarantee that `A` and `B` are the same types
///  - Types should be annotated to ensure type inference does not break
/// the contract by accident
unsafe fn cast_unchecked<A, B>(a: A) -> B {
    let ptr = &*ManuallyDrop::new(a) as *const A as *const B;
    unsafe { ptr.read() }
}
