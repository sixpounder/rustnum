use num_traits::{Float, FloatConst};

use crate::{density, Coord, Shape, Tensor};
use std::vec::Vec;
use std::ops::Range;

/// A normal distribution with `mean` and `scale` parameters in a given `range` of values
pub fn normal<T: 'static + Default + Float + FloatConst>(
    range: Range<T>,
    step: T,
    mean: T,
    scale: T,
) -> Tensor<T> {
    let mut range_vec: Vec<T> = vec![];
    let mut r = range.start;

    while r < range.end {
        range_vec.push(r);
        r = r.add(step);
    }

    let d_size = shape!(range_vec.len());

    let gen = move |_: &Coord, i: u64| -> T {
        density(range_vec[i as usize], mean, scale)
    };

    Tensor::<T>::new(
        d_size,
        Some(&gen),
    )
}

#[cfg(test)]
mod test {
    // use crate::Stats;

    use super::*;
    #[test]
    pub fn normal_distribution() {
        let dist: Tensor<f64> = normal(-5.0..4.9, 0.1, 0.0, 0.2);
        assert_eq!(dist.len(), 100);
    }

    #[test]
    pub fn normal_with_shape() {
        let mut dist = normal(-5.0..4.9, 0.1, 0.0, 0.2);
        assert_eq!(dist.len(), 100);

        dist.reshape(shape!(2, 50));
        assert_eq!(dist.len(), 100);
        // assert_eq!(dist.mean(), 0.0);
        assert_ne!(dist.at(coord!(0, 2)), None);
    }
}
