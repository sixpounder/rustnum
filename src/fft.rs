use crate::prelude::*;
use num_traits::{Float, Num, Pow};
use crate::{shape, Tensor, TensorLike, tensor};

#[derive(Debug)]
pub enum DCTType {
    I,
    II,
    III,
    IIII,
}

pub fn dct(x: Tensor<f64>, dct_type: DCTType) -> Tensor<f64> {
    if x.size() == 0 {
        return tensor!(() => []);
    }

    let mut dct_tensor: Tensor<f64> = Tensor::new_uninit(shape!(x.size()));
    let mut i: usize = 0;
    for component in x.iter() {
        dct_tensor[component.coords] = dct_core(&x, i as u32, &dct_type);
        i += 1;
    }

    dct_tensor
}

pub fn dct_core(x: &Tensor<f64>, k: u32, dct_type: &DCTType) -> f64 {
    match dct_type {
        DCTType::I => dct_1(x, k),
        DCTType::II => dct_2(x, k),
        DCTType::III => dct_3(x, k),
        DCTType::IIII => dct_4(x, k),
    }
}

fn dct_1(x: &Tensor<f64>, k: u32) -> f64 {
    // FORMULA: X[k] = 1 / 2 (x[0] + (-1)^k * x[N-1]) + SUM(n in 1..N-2) { x[n] * cos( (pi/N-1) * n * k ) }
    let pi: f64 = num_traits::FloatConst::PI();
    let x0 = *x.first().unwrap();
    let xlast = *x.last().unwrap();

    // 1 / 2 (x[0] + (-1)^k * x[N-1])
    let first_term = ( x0 + (((-1).pow(k) as f64) * xlast) ) / 2.0;

    // SUM(n in 1..N-2) { x[n] * cos( (pi/N-1) * n * k ) }
    let mut sum_term: f64 = 0.;
    let mut n: f64 = 0.;
    let n_minus_two = x.size() - 2;
    let n_minus_one = (x.size() - 1) as f64;
    for xn in x.iter().take(n_minus_two) {
        sum_term += *xn.value * ((pi / n_minus_one) * (n + 1.) * (k as f64).round()).cos();
        n += 1.0;
    }
    
    first_term + sum_term
}

fn dct_2(x: &Tensor<f64>, k: u32) -> f64 {
    // y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)
    let pi: f64 = num_traits::FloatConst::PI();
    let upper_n = x.size() as f64;
    let pi_over_n = pi / upper_n;

    let mut n: usize = 0;
    let mut sum: f64 = 0.;
    for xn in x.iter() {
        sum += *xn.value * (pi_over_n * (k as f64) * ((n as f64) + 0.5)).cos();
        n += 1;
    }

    sum * 2.
}

fn dct_3<T, K>(x: &Tensor<T>, k: K) -> T
where
    T: Float,
    K: Num,
{
    // FORMULA: X[k] = 1 / 2 (x[0] + (-1)^k * x[N-1]) + SUM(n in 1..N-2) { x[n] * cos( (pi/N-1) * n * k ) }
    todo!()
}

fn dct_4<T, K>(x: &Tensor<T>, k: K) -> T
where
    T: Float,
    K: Num,
{
    // FORMULA: X[k] = 1 / 2 (x[0] + (-1)^k * x[N-1]) + SUM(n in 1..N-2) { x[n] * cos( (pi/N-1) * n * k ) }
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor, coord};

    #[test]
    fn dct1() {
        let values = tensor!((4) => [1., 2., 3., 4.]);
        assert_eq!(values.size(), 4);
        assert_eq!(values.first(), Some(&1.));
        assert_eq!(values.last(), Some(&4.));
        let dct = dct(values, DCTType::I);
        assert_eq!(dct[coord!(0)], 5.5);
    }

    #[test]
    fn dct2() {
        let values = tensor!((4) => [1., 2., 3., 4.]);
        assert_eq!(values.size(), 4);
        assert_eq!(values.first(), Some(&1.));
        assert_eq!(values.last(), Some(&4.));
        let dct = dct(values, DCTType::II);
        assert_eq!(dct[coord!(0)], 20.);
    }
}