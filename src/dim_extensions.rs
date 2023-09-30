use ndarray::{Dim, Dimension, IntoDimension, Ix, IxDyn};

pub trait DimExtension: Dimension {
    fn try_new(ix: &[Ix]) -> Option<Self>;
}

macro_rules! impl_dim_extension {
    ($dim:ty) => {
        impl DimExtension for Dim<$dim> {
            #[inline]
            fn try_new(ix: &[Ix]) -> Option<Self> {
                Some(Dim::<$dim>(ix.try_into().ok()?))
            }
        }
    };
}

impl_dim_extension!([Ix; 0]);
impl_dim_extension!([Ix; 1]);
impl_dim_extension!([Ix; 2]);
impl_dim_extension!([Ix; 3]);
impl_dim_extension!([Ix; 4]);
impl_dim_extension!([Ix; 5]);
impl_dim_extension!([Ix; 6]);

impl DimExtension for IxDyn {
    #[inline]
    fn try_new(ix: &[Ix]) -> Option<Self> {
        Some(ix.into_dimension())
    }
}
