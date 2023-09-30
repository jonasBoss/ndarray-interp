use ndarray::{Dimension, IntoDimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

pub trait DimExtension: Dimension {
    /// create the dimension from an iterator without any size checking
    /// when the iterator yields more values than the number of dimensions,
    /// those get ignored. When the iterator yields less values,
    /// the remaining dimensions will have lenght 0
    ///
    /// for dynamic dimensions the dimension will match the number of yielded elements
    fn new<T: Iterator<Item = Ix>>(ix: T) -> Self;
}

macro_rules! impl_dim_extension {
    ($dim:ty) => {
        impl DimExtension for $dim {
            #[inline]
            fn new<T: Iterator<Item = Ix>>(ix: T) -> Self {
                let mut dim = <$dim>::default();
                dim.as_array_view_mut()
                    .into_iter()
                    .zip(ix)
                    .for_each(|(a, b)| *a = b);
                dim
            }
        }
    };
}

impl_dim_extension!(Ix0);
impl_dim_extension!(Ix1);
impl_dim_extension!(Ix2);
impl_dim_extension!(Ix3);
impl_dim_extension!(Ix4);
impl_dim_extension!(Ix5);
impl_dim_extension!(Ix6);

impl DimExtension for IxDyn {
    #[inline]
    fn new<T: Iterator<Item = Ix>>(ix: T) -> Self {
        ix.collect::<Vec<_>>().into_dimension()
    }
}

#[cfg(test)]
mod test {
    use super::DimExtension;
    use ndarray::{Dimension, Ix0, Ix1, Ix2, Ix3, IxDyn};

    #[test]
    fn create_ix0_short_iter() {
        let ix = Ix0::new(vec![].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), []);
    }

    #[test]
    fn create_ix0_long_iter() {
        let ix = Ix0::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), []);
    }

    #[test]
    fn create_ix1_short_iter() {
        let ix = Ix1::new(vec![].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), [0]);
    }

    #[test]
    fn create_ix1_long_iter() {
        let ix = Ix1::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), [1]);
    }

    #[test]
    fn create_ix2_short_iter() {
        let ix = Ix2::new(vec![].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), [0, 0]);
    }

    #[test]
    fn create_ix2_long_iter() {
        let ix = Ix2::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), [1, 2]);
    }

    #[test]
    fn create_ix3_short_iter() {
        let ix = Ix3::new(vec![].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), [0, 0, 0]);
    }

    #[test]
    fn create_ix3_long_iter() {
        let ix = Ix3::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), [1, 2, 3]);
    }

    #[test]
    fn create_ixdyn_short_iter() {
        let ix = IxDyn::new(vec![].into_iter());
        assert_eq!(ix.as_array_view().as_slice().unwrap(), []);
    }

    #[test]
    fn create_ixdyn_long_iter() {
        let ix = IxDyn::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter());
        assert_eq!(
            ix.as_array_view().as_slice().unwrap(),
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
    }
}
