mod shape;
#[macro_export]
macro_rules! shape {
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

mod tensors;

pub use tensors::*;
pub use shape::*;
