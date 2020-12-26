use crate::{density, Coord, Shape, Tensor};
use std::vec::Vec;
use std::ops::Range;

pub fn normal(
    range: Range<f64>,
    step: f64,
    mean: f64,
    scale: f64,
) -> Tensor<f64> {
    let mut range_vec: Vec<f64> = vec![];
    let mut r = range.start;

    while r < range.end {
        range_vec.push(r);
        r += step;
    }

    let d_size = shape!(range_vec.len());

    let distribution: Tensor<f64> = Tensor::<f64>::new(
        &d_size,
        Some(&move |_: &Coord, i: u64| -> f64 {
            density(range_vec[i as usize], mean, scale)
        }),
    );

    distribution
}

pub fn normal_f64(range: Range<f64>) -> Tensor<f64> {
    normal(range, 0.1, 1.0, 0.5)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    pub fn normal_distribution() {
        let dist: Tensor<f64> = normal(-5.0..4.9, 0.1, 0.0, 0.2);
        // for item in dist.iter() {
        //     println!("{} -> {}", item.coords, item.value);
        // }
        assert_eq!(dist.len(), 100);
    }
}
