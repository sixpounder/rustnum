use crate::{Shape, Tensor};
use std::option::Option;
use std::vec::Vec;

pub fn normal<T>(mean: T, scale: T, shape: Option<Shape>)
where
    T: Default + Copy,
{
    let d_size = shape.unwrap_or(shape![10]);

    let mut distribution: Tensor<T> = Tensor::<T>::new(&d_size, None);
}
