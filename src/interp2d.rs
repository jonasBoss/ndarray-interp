use std::{
    fmt::Debug,
    ops::Sub,
};

use ndarray::{Array1, ArrayBase, Data, Dimension, Ix1, OwnedRepr};
use num_traits::{Num, NumCast, cast};

use crate::BuilderError;

use self::strategies::{Biliniar, StrategyBuilder};

mod strategies;

#[derive(Debug)]
pub struct Interp2D<Sd, Sx, Sy, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension,
{
    x: ArrayBase<Sx, Ix1>,
    y: ArrayBase<Sy, Ix1>,
    data: ArrayBase<Sd, D>,
    strategy: Strat,
}

impl<Sd, D> Interp2D<Sd, OwnedRepr<Sd::Elem>, OwnedRepr<Sd::Elem>, D, Biliniar>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    D: Dimension,
{
    pub fn builder(
        data: ArrayBase<Sd, D>,
    ) -> Interp2DBuilder<Sd, OwnedRepr<Sd::Elem>, OwnedRepr<Sd::Elem>, D, Biliniar> {
        Interp2DBuilder::new(data)
    }
}

#[derive(Debug)]
pub struct Interp2DBuilder<Sd, Sx, Sy, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension,
{
    x: ArrayBase<Sx, Ix1>,
    y: ArrayBase<Sy, Ix1>,
    data: ArrayBase<Sd, D>,
    strategy: Strat,
}

impl<Sd, D> Interp2DBuilder<Sd, OwnedRepr<Sd::Elem>, OwnedRepr<Sd::Elem>, D, Biliniar>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    D: Dimension,
{
    pub fn new(data: ArrayBase<Sd, D>) -> Self {
        let x = Array1::from_iter((0..data.shape()[0]).map(|i| cast(i).unwrap_or_else(||unimplemented!("casting from usize to a number should always work"))));
        let y = Array1::from_iter((0..data.shape()[1]).map(|i| cast(i).unwrap_or_else(||unimplemented!("casting from usize to a number should always work"))));
        Interp2DBuilder {
            x,
            y,
            data,
            strategy: Biliniar,
        }
    }
}

impl<Sd, Sx, Sy, D, Strat> Interp2DBuilder<Sd, Sx, Sy, D, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension, 
{
    pub fn strategy<NewStrat: StrategyBuilder<Sd, Sx, Sy, D>>(self, strategy: NewStrat) -> Interp2DBuilder<Sd, Sx, Sy, D, NewStrat>{
        let Interp2DBuilder { x, y, data, .. } = self;
        Interp2DBuilder { x, y, data, strategy }
    }

    pub fn x<NewSx: Data<Elem = Sd::Elem>>(self, x: ArrayBase<NewSx, Ix1>) -> Interp2DBuilder<Sd, NewSx, Sy, D, Strat>{
        let Interp2DBuilder { y, data, strategy, .. } = self;
        Interp2DBuilder { x, y, data, strategy }
    }

    pub fn y<NewSy: Data<Elem = Sd::Elem>>(self, y: ArrayBase<NewSy, Ix1>) -> Interp2DBuilder<Sd, Sx, NewSy, D, Strat>{
        let Interp2DBuilder { x, data, strategy, .. } = self;
        Interp2DBuilder { x, y, data, strategy }
    }

    pub fn build(self) -> Result<Interp2D<Sd, Sx, Sy, D, Strat>, BuilderError>{
        todo!()
    }
}