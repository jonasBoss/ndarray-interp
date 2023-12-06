# 0.5.0
 - `QubicSpline` changes
  - Move `QubicSpline` interpolator to `interp1d::cubic_spline` module
  - add extrapolation
  - add not-a-knot boundary condition
  - add clamped boundary condition
  - make not-a-knot boundary condition the default
  - allow any first or second derivative as boundary condition

# 0.4.1
 - major performance improvement for `interp_scalar()` methods ~-50%
 - keywords and categorys in crate metadata

# 0.4.0
 - performance improvement for `Interp2D` (`Interp2D::index_point()` is now much faster) -50% for most interpolation methods
 - runtime specialization for `interp_array` methods when query dimension is `Ix1` ~-20% improvement
 - add `interp_into()` and `interp_array_into()` methods for interpolating into a user provided `ArrayViewMut`
 this can improve performance by avoiding memory allocations.
 - fix crash when `IxDyn` dimensions are involved.

# 0.3.2
performance improvement for `VectorExtensions::get_lower_index`.
From -24% for evenly spaced values up to 72% for randomized and 
logarithmic spaced values.

# 0.3.1
 - added type aliases for common interpolators
 - make the `VectorExtensions` trait public
 - add `Interp1D::new_unchecked` and `Interp2D::new_unchecked` methods
 - add bilinear extrapolation
 - impl `Default` for interpolation strategies

# 0.3.0
 - add 2d interpolation
 - add bilinear interpolation strategy
 - rename `Strategy` to `Interp1DStrategy` and `StrategyBuilder` to `Interp1DStrategyBuilder`
 - make extrapolate filed of `Linear` private add `extrapolate(bool)` method instead.

# 0.2.1
 - change interp_array such that it can be called with any 
   kind of array repreresenation (owned, view, ...) technically this 
   breaks public API, but due to type inference this should never manifest 
   as a breaking change.

# 0.2.0
 - updated package structure
 - replaced Interp1DStrategy enum with individual structs
 - added Strategy and StrategyBuilder trait
 - added QubicSpline strategy
 - added traits for custom strategies

# 0.1.1
 - updated package metadata

# 0.1.0
 - Initial release with support for 1d linear interpolation
