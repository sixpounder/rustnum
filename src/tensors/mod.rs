mod shape;

#[macro_export]
/// The following yields a tensor of shape (2, 2, 3)
/// ```
/// tensor![
///     [
///         [1, 2, 3],
///         [4, 5, 6]
///     ],
///     [
///         [7, 8, 9],
///         [10, 11, 12]
///     ]
/// ]
/// ```
macro_rules! tensor {
    ($x:expr) => {
        // Scalar
        Tensor::scalar()
    };
    [ $($x:expr),* ] => {
        Tensor::new()
    };
}

#[macro_export]
macro_rules! shape {
    () => {
        Shape::empty()
    };
    ( $( $x:expr ),* ) => {
        {
            let mut dims = Vec::<usize>::new();
            $(
                dims.push($x);
            )*
            let o = Shape::new(dims);
            o
        }
    };
}

mod coord;
#[macro_export]
macro_rules! coord {
    ( $( $x:expr ),* ) => {
        {
            let mut dims = Vec::<usize>::new();
            $(
                dims.push($x);
            )*
            let o = Coord::new(dims);
            o
        }
    };
}

mod tensors;

pub use tensors::*;
pub use shape::*;
pub use coord::*;
