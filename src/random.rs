use crate::Shape;
use crate::types::TData;
use std::option::Option;
use std::vec::Vec;

pub fn normal<T>(mean: T, scale: T, size: Option<Shape>) {
    let d_size = size.unwrap_or(shape![10]);

    let mut distribution = Vec::<TData>::new();
    let mut dimension = 0;

    while dimension < d_size.len() {
        dimension += 1;
    }
    
}
