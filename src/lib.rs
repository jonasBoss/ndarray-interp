// Copyright (c) 2023 Jonas Bosse
//
// Licensed under the MIT license

//! The ndarray-interp crate provides interpolation algorithms
//! for interpolating _n_-dimesional data.
//!
//! [interp1d::Interp1D] provides functionality to interpolate _n_-dimensional
//! arrays along the first axis. The documentation of the different
//! interpolation methods provide examples.
//!
//! # Custom interpolation strategy
//! This crate defines the traits [`StrategyBuilder`](interp1d::StrategyBuilder)
//! and [`Strategy`](interp1d::Strategy) that allow implementation of user
//! defined interpolation algorithms.
//! see the `custom_strategy.rs` example.
//!

use thiserror::Error;

pub mod interp1d;
pub mod interp2d;
mod vector_extensions;

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
}
