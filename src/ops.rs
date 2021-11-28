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

impl<T> Dot<Vec<T>> for Vec<T>
where
    T: Num + Copy,
{
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

macro_rules! impl_factorial {
    ( $( $t:ty ),* ) => {
        $(
            impl Factorial for $t {
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
        )*
    };
}

impl_factorial!(u8, u16, u32, u64, i8, i16, i32, i64);
