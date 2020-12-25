use crate::{Coord, Shape, Tensor, density};
use std::vec::Vec;
use std::{ops::Range, option::Option};

pub fn normal(range: Range<f64>, step: f64, mean: f64, scale: f64, shape: Option<Shape>) -> Tensor<f64> {
    let d_size = shape.unwrap_or(shape![10]);
    let mut range_vec: Vec<f64> = vec![];
    let mut r = range.start;

    while r < range.end {
        range_vec.push(r);
        r += step;
    }

    let distribution: Tensor<f64> = Tensor::<f64>::new(&d_size, Some(& move |_: &Coord, i: u64| -> f64 {
        // println!("{}", range_vec[i as usize]);
        let d = density(range_vec[i as usize], mean, scale);
        println!("{}", d);
        d
    }));

    distribution
}

pub fn normal_f64(range: Range<f64>) -> Tensor<f64> {
    normal(range, 0.1, 1.0, 0.5, None)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    pub fn test_normal() {
        println!("{:?}", normal(0.0..10.0, 0.1, 0.1, 2.0, Some(shape!(100))));
    }
}
