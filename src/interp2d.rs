use std::{
    fmt::Debug,
    ops::Sub,
};

use ndarray::{Array1, ArrayBase, Data, Dimension, Ix1, OwnedRepr};
use num_traits::{Num, NumCast, cast};

use crate::{
    vector_extensions::{Monotonic, VectorExtensions},
    BuilderError, InterpolateError,
};

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
    Strat: StrategyBuilder<Sd, Sx, Sy, D>
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

    pub fn build(self) -> Result<Interp2D<Sd, Sx, Sy, D, Strat::FinishedStrat>, BuilderError>{
        use self::Monotonic::*;
        use BuilderError::*;
        let Interp2DBuilder { x, y, data, strategy: stratgy_builder } = self;
        if data.ndim() < 2{
            return Err(DimensionError("data dimension needs to be at least 2".into()));
        }
        if data.shape()[0] < Strat::MINIMUM_DATA_LENGHT {
            return Err(NotEnoughData(format!("The 0-dimension has not enough data for the chosen interpolation strategy. Provided: {}, Reqired: {}", data.shape()[0], Strat::MINIMUM_DATA_LENGHT)));
        }
        if data.shape()[1] < Strat::MINIMUM_DATA_LENGHT {
            return Err(NotEnoughData(format!("The 1-dimension has not enough data for the chosen interpolation strategy. Provided: {}, Reqired: {}", data.shape()[1], Strat::MINIMUM_DATA_LENGHT)));
        }
        if !matches!(x.monotonic_prop(), Rising { strict: true }){
            return  Err(Monotonic("The x-axis needs to be strictly monotonic rising".into()));
        }
        if !matches!(y.monotonic_prop(), Rising { strict: true }){
            return Err(Monotonic("The y-axis needs to be strictly monotonic rising".into()));
        }
        if x.len() != data.shape()[0] {
            return Err(AxisLenght(format!("Lenghts of x-axis and data-0-axis need to match. Got x: {}, data-0: {}", x.len(), data.shape()[0])));
        }  
        if y.len() != data.shape()[1]{
            return Err(AxisLenght(format!("Lenghts of y-axis and data-1-axis need to match. Got y: {}, data-1: {}", y.len(), data.shape()[1])));
        }
        
        let strategy = stratgy_builder.build(&x, &y, &data)?;
        Ok(Interp2D { x, y, data, strategy })
    }
}