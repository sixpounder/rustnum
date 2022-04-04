use crate::prelude::*;
use crate::ops::{BinomialTerm, Factorial};
use crate::{activations, TensorLike};
use crate::{coord, generators::density, shape, Tensor};
use num_traits::{Float, FloatConst, NumCast};
use std::ops::Range;
use std::{ops::Add, vec::Vec};

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

    let gen = move |_: Coord, i: usize| -> T { density(range_vec[i], mean, scale) };

    Tensor::new(d_size, gen)
}

/// Generates a uniform distribution from start to end of `range` with each value incremented
/// by `step`
/// # Example
/// ```
/// # use std::ops::{Add, Range};
/// # use rustnum::distributions::arange;
/// # use rustnum::{Tensor, shape, Shape, TensorLike};
/// let mut ranged_values: Tensor<f64> = arange(0.0..9.9, 0.1);
/// // Tensor [0.0, 0.1, 0.2 .... 100.0]
/// assert_eq!(ranged_values.size(), 100);
/// ranged_values.reshape(shape!(10, 2, 5));
/// // Tensor [[[0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9]], [[ .... ]]]
/// assert_eq!(ranged_values.shape(), shape!(10, 2, 5));
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

    Tensor::from(range_vec)
}

/// The Poisson distribution is the discrete probability distribution of the number of events
/// occurring in a given time period, given the average number of times the event occurs over
/// that time period.
///
/// The returned value is a Tensor with a flat shape
///
/// # Conditions of use
/// Conditions for Poisson Distribution:
/// * An event can occur any number of times during a time period.
/// * Events occur independently. In other words, if an event occurs, it does not affect
///   the probability of another event occurring in the same time period.
/// * The rate of occurrence is constant; that is, the rate does not change based on time.
/// * The probability of an event occurring is proportional to the length of the time period.
///   For example, it should be twice as likely for an event to occur in a 2 hour time
///   period than it is for an event to occur in a 1 hour period.
///
/// # Example
/// ```
/// # use rustnum::{Tensor, shape, Shape, coord, Coord, TensorLike};
/// use rustnum::distributions;
/// let dist = distributions::poisson(0..5, 1, 2.5);
/// assert_eq!(dist.size(), 5);
/// assert_eq!(dist[coord!(0)], 0.0820849986238988);
/// assert_eq!(dist[coord!(1)], 0.205212496559747);
/// assert_eq!(dist[coord!(2)], 0.25651562069968376);
/// assert_eq!(dist[coord!(3)], 0.21376301724973645);
/// assert_eq!(dist[coord!(4)], 0.13360188578108528);
/// ```
///
pub fn poisson(range: Range<i32>, step: i32, expected: f64) -> Tensor<f64> {
    let d_size = range.end - range.start;

    let mut distribution: Tensor<f64> = Tensor::new_uninit(shape!(d_size as usize));

    let mut x = range.start;
    let mut i = 0;
    while x < range.end {
        let k: u64 = x as u64;
        let value: f64 = (expected.powi(x) * (-expected).exp()) / k.factorial() as f64;
        distribution.set(coord!(i), value).unwrap();
        i += 1;
        x += step;
    }

    distribution
}

/// A Bernoulli distribution is a discrete distribution with only two
/// possible values for the random variable
pub fn bernoulli(range: Range<i32>, step: i32, p: f64) -> Tensor<f64> {
    let d_size = range.end - range.start;

    let mut distribution: Tensor<f64> = Tensor::new_uninit(shape!(d_size as usize));

    let mut x = range.start;
    let mut i = 0;
    while x < range.end {
        let one_minus_p: f64 = 1.0 - p;
        let value = p.powi(x) * one_minus_p.powi(1 - x);
        distribution.set(coord!(i), value).unwrap();
        i += 1;
        x += step;
    }

    distribution
}

fn binomial_core(n: u64, k: u64, p: f64) -> f64 {
    let one_minus_p: f64 = 1.0 - p;
    let binomial_term: f64 = (n, k).binomial_term() as f64;
    let p_to_k = p.powi(k as i32);
    let one_minus_p_to_n_minus_k = one_minus_p.powi((n - k) as i32);
    let value = p_to_k * one_minus_p_to_n_minus_k * binomial_term;

    value
}

/// The binomial distribution with parameters `n` and `p` is the discrete probability distribution
/// of the number of successes in a sequence of `n` independent experiments, each asking
/// a yes–no question, and each with its own Boolean-valued outcome: success (with probability `p`)
/// or failure (with probability `q = 1 − p`).
/// # Example
/// ```
/// # use rustnum::distributions::binomial;
/// # use rustnum::{Tensor, shape, Shape, coord, Coord, TensorLike};
/// let dist = binomial(5..7, 20, 0.3);
/// assert_eq!(dist.size(), 2);
/// assert_eq!(dist[coord!(0)], 0.1788630505698795);
pub fn binomial(range: Range<u32>, n: u64, p: f64) -> Tensor<f64> {
    let d_size = range.end - range.start;

    let mut distribution: Tensor<f64> = Tensor::new_uninit(shape!(d_size as usize));

    let mut k = range.start;
    let mut i = 0;
    while k < range.end {
        distribution
            .set(coord!(i), binomial_core(n, k as u64, p))
            .unwrap();
        i += 1;
        k += 1;
    }

    distribution
}

fn geometric_core(p: f64, x: u64) -> f64 {
    (1.0 - p).powi(x as i32) * p
}

/// The geometric distribution describes the probability of experiencing a certain amount
/// of failures before experiencing the first success in a series of Bernoulli trials.
/// # Example
/// ```
/// # use rustnum::distributions::geometric;
/// # use rustnum::{Tensor, shape, Shape, coord, Coord, TensorLike};
/// let dist = geometric(0..4, 0.5);
/// assert_eq!(dist.size(), 4);
/// assert_eq!(dist[coord!(0)], 0.5);
/// assert_eq!(dist[coord!(1)], 0.25);
/// assert_eq!(dist[coord!(2)], 0.125);
/// assert_eq!(dist[coord!(3)], 0.0625);
/// ```
pub fn geometric(range: Range<u64>, p: f64) -> Tensor<f64> {
    let d_size = range.end - range.start;

    let mut distribution: Tensor<f64> = Tensor::new_uninit(shape!(d_size as usize));
    let mut k = range.start;
    let mut i = 0;
    while k < range.end {
        distribution.set(coord!(i), geometric_core(p, k)).unwrap();
        i += 1;
        k += 1;
    }

    distribution
}

/// Rectified Linear Unit distribution with `n_elements` values evenly spaced in `range`
pub fn relu<N: Float + FloatConst>(range: Range<N>, n_elements: usize) -> Tensor<N> {
    let step = (range.end - range.start) / NumCast::from(n_elements).unwrap();
    let mut distribution: Tensor<N> = Tensor::new_uninit(shape!(n_elements));
    let mut i = 0;
    let mut last_x = range.start;
    while i < n_elements {
        let value = last_x + step;
        distribution
            .set(
                coord!(i),
                activations::relu(NumCast::from(value).expect("Cannot cast value to correct type")),
            )
            .expect("Cannot evaluate relu on passed value");
        i += 1;
        last_x = value;
    }

    distribution
}

/// Leaky Rectified Linear Unit distribution with `n_elements` values evenly spaced in `range`
pub fn leaky_relu<N: Float + FloatConst>(range: Range<N>, n_elements: usize) -> Tensor<N> {
    let step = (range.end - range.start) / NumCast::from(n_elements).unwrap();
    let mut distribution: Tensor<N> = Tensor::new_uninit(shape!(n_elements));
    let mut i = 0;
    let mut last_x = range.start;
    while i < n_elements {
        let value = last_x + step;
        distribution
            .set(
                coord!(i),
                activations::leaky_relu(
                    NumCast::from(value).expect("Cannot cast value to correct type"),
                ),
            )
            .expect("Cannot evaluate leaky_relu on passed value");
        i += 1;
        last_x = value;
    }

    distribution
}

/// Sigmoid distribution with `n_elements` values evenly spaced in `range`
pub fn sigmoid<N: Float + FloatConst>(range: Range<N>, n_elements: usize) -> Tensor<N> {
    let step = (range.end - range.start) / NumCast::from(n_elements).unwrap();
    let mut distribution: Tensor<N> = Tensor::new_uninit(shape!(n_elements));
    let mut i = 0;
    let mut last_x = range.start;
    while i < n_elements {
        let value = last_x + step;
        distribution
            .set(
                coord!(i),
                activations::sigmoid(
                    NumCast::from(value).expect("Cannot cast value to correct type"),
                ),
            )
            .expect("Cannot evaluate sigmoid on passed value");
        i += 1;
        last_x = value;
    }

    distribution
}

#[cfg(test)]
mod test {
    use crate::TensorLike;

    use super::*;
    #[test]
    fn range_distribution() {
        let dist: Tensor<f64> = arange(-5.0..4.9, 0.1);
        assert_eq!(dist.size(), 100);
    }

    #[test]
    fn reshape_distribution() {
        let mut dist: Tensor<f64> = arange(-5.0..4.9, 0.1);
        assert_eq!(dist.size(), 100);
        dist.reshape(shape!(20, 5));
        assert_eq!(dist.shape_ref(), &shape!(20, 5));
    }

    #[test]
    fn normal_distribution() {
        let dist: Tensor<f64> = normal(-5.0..4.9, 0.1, 0.0, 0.2);
        assert_eq!(dist.size(), 100);
    }

    #[test]
    fn large_normal_distribution() {
        let rng = -100000.0..100000.0;
        let dist: Tensor<f64> = normal(rng, 1., 0.0, 0.2);
        assert_eq!(dist.size(), 200000);
    }

    #[test]
    fn normal_with_shape() {
        let mut dist = normal(-5.0..4.9, 0.1, 0.0, 0.2);
        assert_eq!(dist.size(), 100);

        dist.reshape(shape!(2, 50));
        assert_eq!(dist.size(), 100);
        // assert_eq!(dist.mean(), 0.0);
        assert_ne!(dist.at(coord!(0, 2)), None);
    }

    #[test]
    fn poisson_distribution() {
        let dist = poisson(0..5, 1, 2.5);
        assert_eq!(dist.size(), 5);
        assert_eq!(dist[coord!(0)], 0.0820849986238988);
        assert_eq!(dist[coord!(1)], 0.205212496559747);
        assert_eq!(dist[coord!(2)], 0.25651562069968376);
        assert_eq!(dist[coord!(3)], 0.21376301724973645);
        assert_eq!(dist[coord!(4)], 0.13360188578108528);
    }

    #[test]
    fn binomial_distribution() {
        let dist = binomial(5..7, 20, 0.3);
        assert_eq!(dist.size(), 2);
        assert_eq!(dist.at(coord!(0)).unwrap(), &0.1788630505698795);
    }

    #[test]
    fn geometric_distribution() {
        let dist = geometric(0..4, 0.5);
        assert_eq!(dist.size(), 4);
        assert_eq!(dist[coord!(0)], 0.5);
        assert_eq!(dist[coord!(1)], 0.25);
        assert_eq!(dist[coord!(2)], 0.125);
        assert_eq!(dist[coord!(3)], 0.0625);
    }

    #[test]
    fn relu_distribution() {
        let dist = relu(-10.0..10.0, 200);
        assert!(dist.iter().take(100).all(|c| c.value == &0.0));
        assert!(dist.iter().skip(101).take(100).all(|c| c.value > &0.0));
    }

    #[test]
    fn leaky_relu_distribution() {
        let dist = leaky_relu(-10.0..10.0, 200);
        assert!(dist.iter().take(100).all(|c| c.value < &0.0));
        assert!(dist.iter().skip(101).take(100).all(|c| c.value > &0.0));
    }
}
