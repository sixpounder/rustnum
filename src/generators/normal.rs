use num_traits::{Float, FloatConst};

pub fn density<'a, T: 'a + Float + FloatConst>(x: T, mean: T, scale: T) -> T {
    let one = T::one();
    let two = one + one;
    let p1: T = T::one() / (two * FloatConst::PI() * scale.powf(two)).sqrt();
    let p2: T = (((x - mean).powf(two)) / (two * scale.powf(two))) * one.neg();
    let p2_exp = p2.exp();

    p1 * p2_exp
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noarmal_density_base() {
        assert_eq!(0.000000012151765699646572, density(4.0, 1.0, 0.5));
    }
}
