# ndarray-interp
A Interpolation crate for usage with the rust [ndarray](https://crates.io/crates/ndarray) crate.

# Features
 - Interpolation of _n_-dimensional data along the first axis
 - Interpolation of owned arrays and array views
 - Interpolation at multiple points at once

## Interpolation strategies
 - Linear interpolation with, and without extrapolation
 - Cubic spline interpolation
 - support for custom strategies

## Planned Features
 - More interpolation strategies
 - Interpolation along 2 axis (2D-Interpolation)
 - [rayon](https://crates.io/crates/rayon) support
