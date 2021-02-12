use num_traits::{Float, Num, One, Signed, Zero, float::FloatCore};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Representation of coordinates on a complex plane by a radius and an angle
#[derive(Debug, PartialEq, Clone)]
pub struct PolarCoordinate<T> {
    r: T,
    theta: T,
}

impl<T: Copy> PolarCoordinate<T> {
    pub fn r(&self) -> T {
        self.r
    }

    pub fn theta(&self) -> T {
        self.theta
    }
}

impl<T: Float> PolarCoordinate<T> {
    pub fn to_cartesian(&self) -> Complex<T> {
        Complex::from_polar(self.r, self.theta)
    }
}

impl<T: std::fmt::Display> Display for PolarCoordinate<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}e^i{}", self.r, self.theta)
    }
}

/// Represents a complex number in cartesian coordinates
/// # Example
/// ```
/// # use rustnum::Complex;
/// let c1 = Complex::new(4.0, 2.1);
/// ```
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Complex<T: Num> {
    re: T,
    im: T,
}


impl<T: Clone + Signed> Complex<T> {
    /// Returns the L1 norm `|re| + |im|` -- the [Manhattan distance] from the origin.
    ///
    /// [Manhattan distance]: https://en.wikipedia.org/wiki/Taxicab_geometry
    #[inline]
    pub fn l1_norm(&self) -> T {
        self.re.abs() + self.im.abs()
    }
}

impl<T: Float> One for Complex<T> {
    fn one() -> Self {
        Self {
            re: T::one(),
            im: T::zero(),
        }
    }
}

impl<T: Num> Zero for Complex<T> {
    fn zero() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        return self.re.is_zero() && self.im.is_zero()
    }
}

impl<T> Complex<T>
where
    T: Num,
{
    /// Creates a new complex number from cartesian coordinates
    /// # Example
    /// ```
    /// # use rustnum::Complex;
    /// let c1 = Complex::new(4.0, 2.1);
    /// ```
    pub fn new(real: T, immaginary: T) -> Self {
        Self {
            re: real,
            im: immaginary,
        }
    }

    /// Returns the immaginary unit as a complex number
    pub fn i() -> Self {
        Self {
            re: T::zero(),
            im: T::one(),
        }
    }

    pub fn real(self) -> T {
        self.re
    }

    pub fn immaginary(self) -> T {
        self.im
    }

    pub fn to_tuple(self) -> (T, T) {
        (self.re, self.im)
    }
}

impl<T> Complex<T>
where
    T: Float
{
    /// Creates a complex number from its polar coordinate representation (a modulus length and an angle in radians)
    /// # Example
    /// ```
    /// # use rustnum::Complex;
    /// let c1: Complex<f64> = Complex::from_polar(13.0, 0.39479111969976155);
    /// // 12+i5
    /// ```
    #[inline]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Gets this complex number polar coordinate form
    /// # Example
    /// ```
    /// # use rustnum::Complex;
    /// let c1: Complex<f64> = Complex::new(12.0, 5.0);
    /// let p1 = c1.to_polar();
    /// // 13e^i0.39479111969976155
    /// ```
    #[inline]
    pub fn to_polar(&self) -> PolarCoordinate<T> {
        PolarCoordinate {
            r: self.norm(),
            theta: self.arg(),
        }
    }

    /// Calculate the principal Arg of self.
    #[inline]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// Calculate |self|
    #[inline]
    pub fn norm(self) -> T {
        self.re.hypot(self.im)
    }

    /// The complex conjugate of the complex number `z = x + yi` is given by `x − yi`.
    ///
    /// It is usually denoted by `z*`. This unary operation on complex numbers cannot be expressed by applying
    /// only their basic operations addition, subtraction, multiplication and division.
    /// Geometrically, `z*` is the "reflection" of `z` about the real axis. Conjugating twice gives the original complex number.
    /// # Example
    /// ```
    /// # use rustnum::Complex;
    /// let c1: Complex<f64> = Complex::new(12.0, 5.0);
    /// assert_eq!(c1.conjugate(), Complex::new(12.0, -5.0));
    /// ```
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Returns the square of the norm (since `T` doesn't necessarily
    /// have a sqrt function), i.e. `re^2 + im^2`.
    #[inline]
    pub fn norm_sqr(self) -> T {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im
    }

    /// Computes 1 / self
    /// # Example
    /// ```
    /// # use rustnum::Complex;
    /// let c = Complex::new(12.0, 5.0);
    /// assert_eq!(c.inv(), Complex::new(0.07100591715976332, -0.029585798816568046));
    /// ```
    #[inline]
    pub fn inv(self) -> Self {
        let norm_sqr = self.norm_sqr();
        Self::new(self.re / norm_sqr.clone(), -self.im / norm_sqr)
    }

    /// Multiplicates two complex numbers. Used for the std::ops::Mul implementation.
    #[inline]
    pub fn complex_mul(&self, other: Self) -> Self {
        // c1 * c2 = (c1.real * c2.real - c1.imm * c2.imm) + (c1.real * c2.imm + c1.imm * c2.real)i
        Self {
            re: (self.re.clone() * other.re.clone()) - (self.im.clone() * other.im.clone()),
            im: (self.re.clone() * other.im) + (self.im.clone() * other.re),
        }
    }

    /// Raises `self` to a complex power.
    #[inline]
    pub fn powc(self, exp: Self) -> Self {
        // formula: x^y = (a + i b)^(c + i d)
        // = (ρ e^(i θ))^c (ρ e^(i θ))^(i d)
        //    where ρ=|x| and θ=arg(x)
        // = ρ^c e^(−d θ) e^(i c θ) ρ^(i d)
        // = p^c e^(−d θ) (cos(c θ)
        //   + i sin(c θ)) (cos(d ln(ρ)) + i sin(d ln(ρ)))
        // = p^c e^(−d θ) (
        //   cos(c θ) cos(d ln(ρ)) − sin(c θ) sin(d ln(ρ))
        //   + i(cos(c θ) sin(d ln(ρ)) + sin(c θ) cos(d ln(ρ))))
        // = p^c e^(−d θ) (cos(c θ + d ln(ρ)) + i sin(c θ + d ln(ρ)))
        // = from_polar(p^c e^(−d θ), c θ + d ln(ρ))
        let polar_coords = self.to_polar();
        Self::from_polar(
            polar_coords.r().powf(exp.re) * (-exp.im * polar_coords.theta()).exp(),
            exp.re * polar_coords.theta() + exp.im * polar_coords.r().ln(),
        )
    }

    #[inline]
    pub fn exp(self) -> Self {
        Self {
            re: self.re.exp() * self.im.cos(),
            im: self.re.exp() * self.im.sin(),
        }
    }

    /// Computes the principal value of natural logarithm of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 0]`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ arg(ln(z)) ≤ π`.
    #[inline]
    pub fn ln(self) -> Self {
        // formula: ln(z) = ln|z| + i*arg(z)
        let polar_coords = self.to_polar();
        Self::new(polar_coords.r().ln(), polar_coords.theta())
    }

    /// Raises `self` to a scalar power
    #[inline]
    pub fn powf(self, exponent: T) -> Self {
        // formula: x^y = (ρ e^(i θ))^y = ρ^y e^(i θ y)
        // = from_polar(ρ^y, θ y)
        let polar_coords = self.to_polar();
        Self::from_polar(
            polar_coords.r().powf(exponent),
            polar_coords.theta() * exponent,
        )
    }

    /// Computes the square root of this complex number
    #[inline]
    pub fn sqrt(self) -> Self {
        if self.im.is_zero() {
            if self.re.is_sign_positive() {
                // simple positive real √r, and copy `im` for its sign
                Self::new(self.re.sqrt(), self.im)
            } else {
                // √(r e^(iπ)) = √r e^(iπ/2) = i√r
                // √(r e^(-iπ)) = √r e^(-iπ/2) = -i√r
                let re = T::zero();
                let im = (-self.re).sqrt();
                if self.im.is_sign_positive() {
                    Self::new(re, im)
                } else {
                    Self::new(re, -im)
                }
            }
        } else if self.re.is_zero() {
            // √(r e^(iπ/2)) = √r e^(iπ/4) = √(r/2) + i√(r/2)
            // √(r e^(-iπ/2)) = √r e^(-iπ/4) = √(r/2) - i√(r/2)
            let one = T::one();
            let two = one + one;
            let x = (self.im.abs() / two).sqrt();
            if self.im.is_sign_positive() {
                Self::new(x, x)
            } else {
                Self::new(x, -x)
            }
        } else {
            // formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
            let one = T::one();
            let two = one + one;
            let polar_coords = self.to_polar();
            Self::from_polar(polar_coords.r().sqrt(), polar_coords.theta() / two)
        }
    }

    /// Computes the sine of `self`.
    #[inline]
    pub fn sin(self) -> Self {
        // formula: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    /// Computes the cosine of `self`.
    #[inline]
    pub fn cos(self) -> Self {
        // formula: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    /// Computes the hyperbolic sine of `self`.
    #[inline]
    pub fn sinh(self) -> Self {
        // formula: sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
        Self::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }

    /// Computes the hyperbolic cosine of `self`.
    #[inline]
    pub fn cosh(self) -> Self {
        // formula: cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
        Self::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }

    #[inline]
    pub fn atan(self) -> Self {
        // formula: arctan(z) = (ln(1+iz) - ln(1-iz))/(2i)
        let i = Self::i();
        let one = Self::one();
        let two = one + one;
        ((one + i * self).ln() - (one - i * self).ln()) / (two * i)
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self {
            re: self.re.abs(),
            im: self.im.abs(),
        }
    }

    #[inline]
    pub fn acos(self) -> Self {
        // formula: arccos(z) = -i ln(i sqrt(1-z^2) + z)
        let i = Self::i();
        -i * (i * (Self::one() - (self * self)).sqrt() + self).ln()
    }

    /// Computes the principal value of the inverse sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Re(asin(z)) ≤ π/2`.
    #[inline]
    pub fn asin(self) -> Self {
        // formula: arcsin(z) = -i ln(sqrt(1-z^2) + iz)
        let i = Self::i();
        -i * ((Self::one() - self * self).sqrt() + i * self).ln()
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    #[inline]
    pub fn is_sign_positive(&self) -> bool {
        self.re.is_sign_positive()
    }
}

impl<T: FloatCore> Complex<T> {
    /// Checks if the given complex number is NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Checks if the given complex number is infinite
    #[inline]
    pub fn is_infinite(self) -> bool {
        !self.is_nan() && (self.re.is_infinite() || self.im.is_infinite())
    }

    /// Checks if the given complex number is finite
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Checks if the given complex number is normal
    #[inline]
    pub fn is_normal(self) -> bool {
        self.re.is_normal() && self.im.is_normal()
    }
}

impl<T: Float + std::fmt::Display> Display for Complex<T> {
    /// Displays a complex number in the form a+ib
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im.is_sign_negative() {
            write!(f, "{}-i{}", self.re, self.im.abs())
        } else if self.im.is_sign_positive() {
            write!(f, "{}+i{}", self.re, self.im)
        } else {
            write!(f, "{}", self.re)
        }
    }
}

// #region ops

// Complex + Complex
impl<T: Num + Add<Output = T>> Add<Complex<T>> for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Float + Add<Output = T>> AddAssign<Complex<T>> for Complex<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.re = self.re + rhs.re;
        self.im = self.im + rhs.im;
    }
}

// Complex + Real
impl<T: Num + Add<Output = T>> Add<T> for Complex<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::Output {
            re: self.re + rhs,
            im: self.im,
        }
    }
}

// Complex - Complex
impl<T: Num + Sub<Output = T>> Sub<Complex<T>> for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Float + Sub<Output = T>> SubAssign<Complex<T>> for Complex<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.re = self.re - rhs.re;
        self.im = self.im - rhs.im;
    }
}

// Complex - Real
impl<T: Num + Sub<Output = T>> Sub<T> for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::Output {
            re: self.re - rhs,
            im: self.im,
        }
    }
}

// Complex * Complex
impl<T> Mul<Complex<T>> for Complex<T>
where
    T: Float,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.complex_mul(rhs)
    }
}

impl<T: Float + Mul<Output = T>> Mul<T> for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl<T> MulAssign<Complex<T>> for Complex<T>
where
    T: Float,
{
    fn mul_assign(&mut self, rhs: Self) {
        let result = self.complex_mul(rhs);
        self.re = result.re;
        self.im = result.im;
    }
}

impl<T: Float + Mul<Output = T>> MulAssign<T> for Complex<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.re = self.re * rhs;
        self.im = self.im * rhs;
    }
}

impl<T> Div<Complex<T>> for Complex<T>
where
    T: Float,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // self.complex_div(Complex::new(rhs, T::zero()))
        let mul_factor = T::one() / (rhs.re.powi(2) * rhs.im.powi(2));
        Self {
            re: mul_factor * (self.re * rhs.re + self.im * rhs.im),
            im: mul_factor * (self.im * rhs.re + self.re * rhs.im),
        }
    }
}

impl<T> Div<T> for Complex<T>
where
    T: Float,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self {
            re: self.re / rhs,
            im: self.im
        }
    }
}

impl<T: Num + Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::Output {
            re: -self.re,
            im: -self.im,
        }
    }
}

// #endregion

#[cfg(test)]
mod test {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn new() {
        let c: Complex<f64> = Complex::new(3.0, 4.0);
        assert_eq!(c.re, 3.0);
        assert_eq!(c.im, 4.0);
    }

    #[test]
    fn sum() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c + 1.0, Complex { re: 4.0, im: 4.0 });

        let c = Complex::new(3.0, 4.0);
        let c2 = Complex::new(1.0, 0.1);
        assert_eq!(c + c2, Complex { re: 4.0, im: 4.1 });
    }

    #[test]
    fn to_polar() {
        let c = Complex::new(12.0, 5.0).to_polar();
        //13e^i0.39479111969976155
        assert_eq!(c.r(), 13.0);
        assert_eq!(c.theta(), 0.3947911196997615);
    }

    #[test]
    fn from_polar() {
        let c = Complex::from_polar(13.0, 0.3947911196997615);
        //13e^i0.39479111969976155
        assert_eq!(c.re, 12.0);
        assert_eq!(c.im, 5.0);
    }

    #[test]
    fn sub() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c - 1.0, Complex { re: 2.0, im: 4.0 });

        let c = Complex::new(3.0, 4.0);
        let c2 = Complex::new(1.0, 0.1);
        assert_eq!(c - c2, Complex { re: 2.0, im: 3.9 });
    }

    #[test]
    fn mul() {
        let c = Complex::new(3.0, 4.0);
        let c2 = Complex::new(2.0, 5.0);
        assert_eq!(c * c2, Complex::new(-14.0, 23.0));
        assert_eq!(c * 2.0, Complex::new(6.0, 8.0));
    }

    #[test]
    fn exp() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(
            c.exp(),
            Complex::new(-13.128783081462158, -15.200784463067954)
        );
    }

    #[test]
    fn polar_coords() {
        let c = Complex::new(12.0, 5.0);
        let p = c.to_polar();
        assert_eq!(
            p,
            PolarCoordinate {
                r: 13.0,
                theta: 0.3947911196997615
            }
        );
        assert_eq!(Complex::from_polar(p.r, p.theta), Complex::new(12.0, 5.0));
    }

    #[test]
    fn conjugate() {
        let c1: Complex<f64> = Complex::new(12.0, 5.0);
        assert_eq!(c1.conjugate(), Complex::new(12.0, -5.0));
    }

    #[test]
    fn inv() {
        let c = Complex::new(12.0, 5.0);
        assert_eq!(c.inv(), Complex::new(0.07100591715976332, -0.029585798816568046));
    }

    #[test]
    fn repr() {
        let c = Complex::new(12.0, 5.0);
        let mut s = String::new();
        match write!(s, "{}", c) {
            Ok(_) => {
                assert_eq!(s.to_string(), "12+i5");
            }
            Err(_) => {
                panic!("Could not test complex display trait impl (you broke the test)");
            }
        }
    }
}
