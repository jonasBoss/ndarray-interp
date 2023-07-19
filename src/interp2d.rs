use std::{
    fmt::Debug,
    ops::Sub,
};

use ndarray::{Array1, ArrayBase, Data, Dimension, Ix1, OwnedRepr, Ix2, Array, RemoveAxis, DimAdd, NdIndex, Axis, IntoDimension, AxisDescription, Slice};
use num_traits::{Num, NumCast, cast};

use crate::{
    vector_extensions::{Monotonic, VectorExtensions},
    BuilderError, InterpolateError,
};

use self::strategies::{Biliniar, StrategyBuilder, Strategy};

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

impl<Sd, Sx, Sy, Strat> Interp2D<Sd, Sx, Sy, Ix2, Strat>
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    Strat: Strategy<Sd, Sx, Sy, Ix2>,
{
    pub fn interp_scalar(&self, x: Sx::Elem, y: Sy::Elem) -> Result<Sd::Elem, InterpolateError>{
        Ok(*self.interp(x, y)?.first().unwrap_or_else(||unreachable!()))
    }
}

impl<Sd, Sx,Sy,D,Strat> Interp2D<Sd, Sx, Sy, D, Strat> 
where
    Sd: Data,
    Sd::Elem: Num + PartialOrd + NumCast + Copy + Debug + Sub,
    Sx: Data<Elem = Sd::Elem>,
    Sy: Data<Elem = Sd::Elem>,
    D: Dimension + RemoveAxis,
    D::Smaller: RemoveAxis,
    Strat: Strategy<Sd, Sx, Sy, D>,
{
    pub fn interp(&self, x: Sx::Elem, y:Sy::Elem) -> Result<Array<Sd::Elem, <D::Smaller as Dimension>::Smaller>, InterpolateError>{
        let dim = self.data.raw_dim().remove_axis(Axis(0)).remove_axis(Axis(0));
        let mut target = Array::zeros(dim);
        self.strategy
            .interp_into(self, target.view_mut(), x, y)
            .map(|_|target)
    }

    pub fn interp_array<Sqx, Sqy, Dq>(&self, xs: &ArrayBase<Sqx, Dq>, ys: &ArrayBase<Sqy, Dq>
    )-> Result<Array<Sd::Elem, <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output>, InterpolateError>
    where
        Sqx: Data<Elem = Sd::Elem>,
        Sqy: Data<Elem = Sy::Elem>,
        Dq: Dimension,
        Dq: DimAdd<<D::Smaller as Dimension>::Smaller>
    {
        let mut dim = <Dq as DimAdd<<D::Smaller as Dimension>::Smaller>>::Output::default();
        assert!(xs.shape() == ys.shape());
        dim.as_array_view_mut()
            .into_iter()
            .zip(xs.shape().iter().chain(self.data.shape()[2..].iter()))
            .for_each(|(new_axis, &len)|{
                *new_axis = len;
            });
        let mut zs = Array::zeros(dim);
        for (index, &x) in xs.indexed_iter(){
            let current_dim = index.clone().into_dimension();
            let y = *ys.get(current_dim.clone()).unwrap_or_else(||unreachable!());
            let subview = 
                zs.slice_each_axis_mut(|AxisDescription { axis: Axis(nr), .. }| match current_dim
                    .as_array_view()
                    .get(nr)
                {
                    Some(idx) => Slice::from(*idx..*idx + 1),
                    None => Slice::from(..),
                });
            
            self.strategy.interp_into(
                self, 
                subview.into_shape(self.data.raw_dim().remove_axis(Axis(0)).remove_axis(Axis(0)))
                    .unwrap_or_else(|_|unreachable!()), 
                x, 
                y
            )?;
        }

        Ok(zs)
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
