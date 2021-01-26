use crate::{Coord, Shape, Tensor};
use std::ops::{Add, Range};

/// Generates a uniform distribution from start to end of `range` with each value incremented
/// by `step`
/// # Example
/// ```
/// # use std::ops::{Add, Range};
/// # use rustnum::ranges::arange;
/// # use rustnum::{Tensor, shape, Shape};
/// let mut ranged_values: Tensor<f64> = arange(0.0..9.9, 0.1);
/// // Tensor [0.0, 0.1, 0.2 .... 100.0]
/// assert_eq!(ranged_values.len(), 100);
/// ranged_values.reshape(shape!(10, 2, 5));
/// // Tensor [[[0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9], [ .... ]]]
/// assert_eq!(ranged_values.shape(), &shape!(10, 2, 5));
/// ```
pub fn arange<T>(range: Range<T>, step: T) -> Tensor<T>
where
    T: Add<Output = T> + Default + Copy + std::cmp::PartialOrd + 'static,
{
    let mut range_vec: Vec<T> = vec![];
    let mut r = range.start;

    while r < range.end {
        range_vec.push(r);
        r = r + step;
    }

    let d_size = shape!(range_vec.len());

    let distribution: Tensor<T> = Tensor::<T>::new(
        d_size,
        Some(&move |_: &Coord, i: u64| -> T { range_vec[i as usize] }),
    );

    distribution
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn normal_distribution() {
        let dist: Tensor<f64> = arange(-5.0..4.9, 0.1);
        assert_eq!(dist.len(), 100);
    }

    #[test]
    fn reshape_distribution() {
        let mut dist: Tensor<f64> = arange(-5.0..4.9, 0.1);
        assert_eq!(dist.len(), 100);
        dist.reshape(shape!(20, 5));
        assert_eq!(dist.shape(), &shape!(20, 5));
    }
}
