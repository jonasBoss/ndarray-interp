use ndarray::{Ix2, Ix3, OwnedRepr, ViewRepr};

use super::Interp2D;

/// two-dimensional interpolant for owned data
pub type Interp2DOwned<A, D, S> = Interp2D<OwnedRepr<A>, OwnedRepr<A>, OwnedRepr<A>, D, S>;
/// two-dimensional interpolant for data views and axis views
pub type Interp2DView<A, D, S> = Interp2D<ViewRepr<A>, ViewRepr<A>, ViewRepr<A>, D, S>;
/// two-dimensional interpolant for data views and owned axis
pub type Interp2DDataView<A, D, S> = Interp2D<ViewRepr<A>, OwnedRepr<A>, OwnedRepr<A>, D, S>;
/// two-dimensional interpolant for scalar, owned data
pub type Interp2DScalar<A, S> = Interp2DOwned<A, Ix2, S>;
/// two-dimensional interpolant for vectroized, owned data
pub type Interp2DVec<A, S> = Interp2DOwned<A, Ix3, S>;
