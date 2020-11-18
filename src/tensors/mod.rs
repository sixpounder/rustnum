mod shape;
#[macro_export]
macro_rules! shape {
    ( $( $x:expr ),* ) => {
        {
            let mut dims = Vec::<usize>::new();
            $(
                dims.push($x);
            )*
            Shape::new(dims)
        }
    };
}

mod tensors;

pub use tensors::*;
pub use shape::*;
