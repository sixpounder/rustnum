use std::{fmt::Display, ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign}};

use crate::ops::Trigo;

type ComplexBaseType = f64;

/// Representation of coordinates on a complex plane by a modulus and an angle
#[derive(Debug, PartialEq)]
pub struct PolarCoordinate {
    radius: ComplexBaseType,
    angle: ComplexBaseType
}

impl PolarCoordinate {
    pub fn new(radius: ComplexBaseType, angle: ComplexBaseType) -> Self {
        Self {
            radius,
            angle
        }
    }

    pub fn radius(&self) -> ComplexBaseType {
        self.radius
    }

    pub fn angle(&self) -> ComplexBaseType {
        self.angle
    }
}

impl Display for PolarCoordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}e^{}i", self.radius, self.angle)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Complex {
    real: ComplexBaseType,
    immaginary: ComplexBaseType
}

impl Complex {
    pub fn i() -> Self {
        Self {
            real: 0.0,
            immaginary: 1.0
        }
    }

    pub fn new(real: ComplexBaseType, immaginary: ComplexBaseType) -> Self {
        Self {
            real,
            immaginary
        }
    }

    pub fn from_polar(length: ComplexBaseType, angle: ComplexBaseType) -> Self {
        Self {
            real: length * angle.cos(),
            immaginary: length * angle.sin()
        }
    }

    pub fn polar_coords(&self) -> PolarCoordinate {
        PolarCoordinate {
            radius: (self.real.powi(2) + self.immaginary.powi(2)).sqrt(),
            angle: (self.real / self.immaginary).atan()
        }
    }

    pub fn exp(&self) -> Self {
        Complex {
            real: self.real.exp() * self.immaginary.cos(),
            immaginary: self.real.exp() * self.immaginary.sin()
        }
    }

    pub fn square(&self) -> Self {
        self.clone() * self.clone()
    }

    /// The complex conjugate of the complex number `z = x + yi` is given by `x − yi`.
    ///
    /// It is usually denoted by `z*`. This unary operation on complex numbers cannot be expressed by applying
    /// only their basic operations addition, subtraction, multiplication and division.
    /// Geometrically, `z*` is the "reflection" of `z` about the real axis. Conjugating twice gives the original complex number.
    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            immaginary: -self.immaginary
        }
    }

    pub fn complex_mul(&self, other: Self) -> Self {
        // c1 * c2 = (c1.real * c2.real - c1.imm * c2.imm) + (c1.real * c2.imm + c1.imm * c2.real)i
        Self {
            real: (self.real * other.real) - (self.immaginary * other.immaginary),
            immaginary: (self.real * other.immaginary) + (self.immaginary * other.real)
        }
    }

    pub fn complex_div(&self, other: Self) -> Self {
        let mul_factor = 1.0 / (other.real.powi(2) * other.immaginary.powi(2));
        Self {
            real: mul_factor * (self.real * other.real + self.immaginary * other.immaginary),
            immaginary: mul_factor  * (self.immaginary * other.real + self.real * other.immaginary)
        }
    }
}

impl Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.immaginary < 0.0 {
            write!(f, "({}-{}i)", self.real, self.immaginary.abs())
        } else if self.immaginary > 0.0 {
            write!(f, "({}+{}i)", self.real, self.immaginary)
        } else {
            write!(f, "{}", self.real)
        }
    }
}

impl Add<Complex> for Complex {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Self::Output {
        Self::Output {
            real: self.real + rhs.real,
            immaginary: self.immaginary + rhs.immaginary
        }
    }
}

impl AddAssign<Complex> for Complex {
    fn add_assign(&mut self, rhs: Complex) {
        self.real = self.real + rhs.real;
        self.immaginary = self.immaginary + rhs.immaginary;
    }
}

impl Add<ComplexBaseType> for Complex {
    type Output = Complex;

    fn add(self, rhs: ComplexBaseType) -> Self::Output {
        Self::Output {
            real: self.real + rhs,
            immaginary: self.immaginary
        }
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            real: self.real - rhs.real,
            immaginary: self.immaginary - rhs.immaginary
        }
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.real = self.real - rhs.real;
        self.immaginary = self.immaginary - rhs.immaginary;
    }
}

impl Sub<ComplexBaseType> for Complex {
    type Output = Complex;

    fn sub(self, rhs: ComplexBaseType) -> Self::Output {
        Self::Output {
            real: self.real - rhs,
            immaginary: self.immaginary
        }
    }
}

impl Mul<Complex> for Complex {
    type Output = Complex;

    fn mul(self, rhs: Self) -> Self::Output {
        self.complex_mul(rhs)
    }
}

impl Mul<ComplexBaseType> for Complex {
    type Output = Complex;

    fn mul(self, rhs: ComplexBaseType) -> Self::Output {
        Self::Output {
            real: self.real * rhs,
            immaginary: self.immaginary * rhs
        }
    }
}

impl MulAssign<Complex> for Complex {
    fn mul_assign(&mut self, rhs: Complex) {
        let result = self.complex_mul(rhs);
        self.real = result.real;
        self.immaginary = result.immaginary;
    }
}

impl MulAssign<ComplexBaseType> for Complex {
    fn mul_assign(&mut self, rhs: ComplexBaseType) {
        self.real = self.real * rhs;
        self.immaginary = self.immaginary * rhs;
    }
}

impl Div<ComplexBaseType> for Complex {
    type Output = Self;

    fn div(self, rhs: ComplexBaseType) -> Self::Output {
        self.complex_div(Complex::new(rhs, 0.0))
    }
}

impl Div<Complex> for Complex {
    type Output = Self;

    fn div(self, rhs: Complex) -> Self::Output {
        self.complex_div(rhs)
    }
}

impl Neg for Complex {
    type Output = Complex;

    fn neg(self) -> Self::Output {
        Self::Output {
            real: -self.real,
            immaginary: -self.immaginary
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn new() {
        let c: Complex = Complex::new(3.0, 4.0);
        assert_eq!(c.real, 3.0);
        assert_eq!(c.immaginary, 4.0);
    }

    #[test]
    fn sums() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c + 1.0, Complex { real: 4.0, immaginary: 4.0 });

        let c = Complex::new(3.0, 4.0);
        let c2 = Complex::new(1.0, 0.1);
        assert_eq!(c + c2, Complex { real: 4.0, immaginary: 4.1 });
    }

    #[test]
    fn mul() {
        let c = Complex::new(3.0, 4.0);
        let c2 = Complex::new(2.0, 5.0);
        assert_eq!(c * c2, Complex::new(-14.0, 23.0));
    }

    #[test]
    fn exp() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c.exp(), Complex::new(-13.128783081462158, -15.200784463067954));
    }

    #[test]
    fn polar_coords() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c.polar_coords(), PolarCoordinate { radius: 5.0, angle: 0.6435011087932844 });
    }
}
