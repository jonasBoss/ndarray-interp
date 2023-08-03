use ndarray::{OwnedRepr, Ix1, Ix2, ViewRepr, Ix3};

use crate::{interp1d::Interp1D, interp2d::Interp2D};

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

/// two-dimensional interpolant for owned data
pub type Interp2DOwned<A, D, S> = Interp2D<OwnedRepr<A>, OwnedRepr<A>, OwnedRepr<A>, D, S>;
/// two-dimensional interpolant for data views and axis views
pub type Interp2DView<A, D, S> = Interp2D<ViewRepr<A>, ViewRepr<A>, ViewRepr<A>, D, S>;
/// two-dimensional interpolant for data views and owned axis
pub type Interp2DDataView<A, D, S> = Interp2D<ViewRepr<A>, OwnedRepr<A>, OwnedRepr<A>, D, S>;
/// two-dimensional interpolant for scalar, owned data
pub type Interp2DScalar<A,S> = Interp2DOwned<A, Ix2, S>;
/// two-dimensional interpolant for vectroized, owned data
pub type Interp2DVec<A,S> = Interp2DOwned<A, Ix3, S>;
