use crate::types::TData;
use std::option::Option;
use std::vec::Vec;

pub fn normal(mean: TData, scale: TData, size: Option<Vec<usize>>) {
    let d_size = size.unwrap_or(vec![1, 10]);

    let mut distribution = Vec::<TData>::new();
    let mut dimension = 0;

    while dimension < d_size.len() {
        dimension += 1;
    }
    
}
