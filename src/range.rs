use crate::{Coord, Shape, Tensor};
use std::ops::{Add, Range};

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
        &d_size,
        Some(&move |_: &Coord, i: u64| -> T { range_vec[i as usize] }),
    );

    distribution
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    pub fn normal_distribution() {
        let dist: Tensor<f64> = arange(-5.0..4.9, 0.1);
        assert_eq!(dist.len(), 100);
    }
}
