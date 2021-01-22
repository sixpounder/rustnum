pub trait Dot<T> {
    type Output;
    fn dot(&self, rhs: T) -> Self::Output;
}

pub trait Trigo {
    type Output: ?Sized;
    fn sin(&self) -> Self::Output;
    fn cos(&self) -> Self::Output;
}
