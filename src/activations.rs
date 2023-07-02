//! Collection of classic ML activation functions
use num_traits::{Float, FloatConst, Num};

pub fn max<N>(x: N, y: N) -> N where N: Num + PartialOrd {
    if x > y { x } else { y }
}

/// Rectified Linear Unit function
/// 
/// Formula: `y = max(0, x)`
pub fn relu<N>(x: N) -> N where N: Float {
    max(N::zero(), x)
}

/// Leaky Rectified Linear Unit function
/// 
/// Formula: `y = max(0,01 * x, x)`
pub fn leaky_relu<N>(x: N) -> N where N: Float {
    max(N::from(0.01).unwrap() * x, x)
}

/// Sigmoid function
/// 
/// Formula: y = 1 / 1 + e^(-x)
pub fn sigmoid<N>(x: N) -> N where N: Float + FloatConst {
    let e: N = num_traits::float::FloatConst::E();
    N::one() / (N::one() + e.powf(-x))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn activation_sigmoid() {
        assert_eq!(sigmoid(2.0), 0.880_797_077_977_882_3);
    }

    #[test]
    fn activation_relu() {
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
    }

    #[test]
    fn activation_leaky_relu() {
        assert!(leaky_relu(-3.0).abs() > 0.0);
        assert_eq!(leaky_relu(3.0), 3.0);
    }
}
