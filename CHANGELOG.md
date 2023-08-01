# 0.3.0
 - add 2d interpolation
 - add biliniar interpolation strategy
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
