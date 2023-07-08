//! The ndarray-interp crate provides interpolation algorithms
//! for interpolating _n_-dimesional data.
//!
//! [Interp1D] provides functionality to interpolate _n_-dimensional
//! arrays along the first axis. The documentation of the different
//! interpolation methods provides examples.
mod interp1d;
mod interp2d;
mod vector_extensions;

pub use interp1d::*;
pub use interp2d::*;
