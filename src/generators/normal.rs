pub fn density(x: f64, mean: f64, scale: f64) -> f64 {
    let p1: f64 = 1.0 / (2.0 * std::f64::consts::PI * (scale.powf(2.0))).sqrt();
    let p2: f64 = (((x - mean).powf(2.0)) / (2.0 * scale.powf(2.0))) * -1.0;
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
