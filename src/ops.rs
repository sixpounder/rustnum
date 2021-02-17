use num_traits::Num;

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

impl Factorial for i8 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if *self == 0 {
            1
        } else {
            self * (self - 1).factorial()
        }
    }
}

impl Factorial for i16 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if *self == 0 {
            1
        } else {
            self * (self - 1).factorial()
        }
    }
}

impl Factorial for i32 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if *self == 0 {
            1
        } else {
            self * (self - 1).factorial()
        }
    }
}

impl Factorial for i64 {
    fn factorial(&self) -> Self {
        assert!(*self >= 0);
        if *self == 0 {
            1
        } else {
            self * (self - 1).factorial()
        }
    }
}
