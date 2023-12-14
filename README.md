# ndarray-interp
A Interpolation crate for usage with the rust [ndarray](https://crates.io/crates/ndarray) crate.

# Features
 - 1D-Interpolation of _n_-dimensional data along the first axis
 - 2D-Interpolation of _n_-dimensional data along the first two axes
 - Add your own Interpolation algorithms
 - Interpolation of owned arrays and array views
 - Interpolation at multiple points at once

## Interpolation strategies
 - Linear interpolation with, and without extrapolation
 - Cubic spline interpolation [Wikipedia](https://en.wikipedia.org/wiki/Spline_interpolation)
 - Bilinear interpolation with, and without extrapolation [Wikipedia](https://en.wikipedia.org/wiki/Bilinear_interpolation)

## Planned Features
 - More interpolation strategies
 - [rayon](https://crates.io/crates/rayon) support
