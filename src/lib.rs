// Copyright (c) 2023 Jonas Bosse
//
// Licensed under the MIT license

//! The ndarray-interp crate provides interpolation algorithms
//! for interpolating _n_-dimesional data.
//!
//! 1D and 2D interpolation is supported. See the modules [interp1d] and [interp2d]
//!
//! # Custom interpolation strategy
//! This crate defines traits to allow implementation of user
//! defined interpolation algorithms.
//! see the `custom_strategy.rs` example.
//!

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
    /// The lengths of interpolation axis and the
    /// corresponding data axis do not match
    #[error("{0}")]
    AxisLenght(String),
    #[error("{0}")]
    DimensionError(String),
}

/// Errors during Interpolation
#[derive(Debug, Error)]
pub enum InterpolateError {
    #[error("{0}")]
    OutOfBounds(String),
    #[error("{0}")]
    InvalidArguments(String),
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
