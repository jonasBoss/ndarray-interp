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

use ndarray::ShapeError;
use thiserror::Error;

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
    #[error("{1}")]
    ShapeError(ShapeError, String)
}
