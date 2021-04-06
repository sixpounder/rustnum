use num_traits::{Num, Zero};

pub trait Stats {
    type Output;

    fn mean(&self) -> Self::Output;
    fn max(&self) -> Self::Output;
    fn min(&self) -> Self::Output;
}

pub trait Dot<T> {
    type Output;
    fn dot(&self, rhs: T) -> Self::Output;
}

impl<T> Dot<Vec<T>> for Vec<T> where T: Num + Copy {
    type Output = T;

    fn dot(&self, rhs: Vec<T>) -> Self::Output {
        let mut acc: Self::Output = T::zero();
        for i in 0..self.len() {
            acc = acc + (self[i] * rhs[i]);
        }
        acc
    }
}

pub trait Trigo {
    type Output: ?Sized;
    fn sin(&self) -> Self::Output;
    fn cos(&self) -> Self::Output;
}

pub trait BinomialTerm {
    type Output;
    fn binomial_term(self) -> Self::Output;
}

impl<T: Num + Factorial> BinomialTerm for (T, T) {
    type Output = T;

    fn binomial_term(self) -> Self::Output {
        self.0.factorial() / (self.1.factorial() * (self.0 - self.1).factorial())
    }
}

pub trait Factorial {
    fn factorial(&self) -> Self;
}

impl Factorial for u8 {
    fn factorial(&self) -> Self {
        if *self == 0 {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}

impl Factorial for u16 {
    fn factorial(&self) -> Self {
        if *self == 0 {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}

impl Factorial for u32 {
    fn factorial(&self) -> Self {
        if self.is_zero() {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}

impl Factorial for u64 {
    fn factorial(&self) -> Self {
        if *self == 0 {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}

impl Factorial for i8 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if *self == 0 {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}

impl Factorial for i16 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if *self == 0 {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}

impl Factorial for i32 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if self.is_zero() {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}

impl Factorial for i64 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if *self == 0 {
            1
        } else {
            let mut acc: Self = 1;
            for i in 2..=*self {
                acc = acc * i;
            }

            acc
        }
    }
}