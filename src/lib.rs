#[macro_use]
mod tensors;

mod types;
mod generators;
mod random;

pub use tensors::*;

pub fn arange<T>(start: T, end: T, step: T) -> Vec<T>
where
    T: std::cmp::PartialOrd + Copy + std::ops::Add<Output = T>
{
    let mut rng = Vec::<T>::new();
    let mut i = start;
    while i < end {
        rng.push(i);
        i = i + step;
    }

    rng
}
