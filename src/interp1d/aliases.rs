use ndarray::{Ix2, ViewRepr, OwnedRepr, Ix1};

use super::Interp1D;

/// one-dimensional interpolant for owned data
pub type Interp1DOwned<A, D, S> = Interp1D<OwnedRepr<A>, OwnedRepr<A>, D, S>;
/// one-dimensional interpolant for data views and axis views
pub type Interp1DView<A, D, S> = Interp1D<ViewRepr<A>, ViewRepr<A>, D, S>;
/// one-dimensional interpolant for data views and owned axis
pub type Interp1DDataView<A, D, S> = Interp1D<ViewRepr<A>, OwnedRepr<A>, D, S>;
/// one-dimensional interpolant for scalar, owned data
pub type Interp1DScalar<A, S> = Interp1DOwned<A, Ix1, S>;
/// one-dimensional interpolant for vectroized, owned data
pub type Interp1DVec<A, S> = Interp1DOwned<A, Ix2, S>;
