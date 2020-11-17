use crate::types::Shape;
use crate::types::TData;

type TensorValueGenerator = dyn Fn(&Shape) -> TData;

fn makeTensor(shape: Shape, generator: &TensorValueGenerator, at_coords: Option<&Shape>) -> Tensor {
    let mut out_tensor = Tensor {
        values: vec![],
        axis: vec![],
    };

    if shape.len() > 1 {
        for axis in 0..shape[0] {
            out_tensor.axis.push(Box::new(makeTensor(vec![*shape.last().unwrap()], generator, at_coords)));
        }
    } else {
        out_tensor.values.push(generator(at_coords.unwrap()));
    }

    out_tensor
}


pub struct Tensor {
    values: Vec<TData>,
    axis: Vec<Box<Tensor>>,
}

impl Tensor {
    pub fn new(shape: Shape, generator: Option<&TensorValueGenerator>) -> Self {
        makeTensor(shape, generator.unwrap_or(&|_| 0.0 ), None)
    }

    /// Gets the shape of this tensor
    pub fn shape(&self) {}

    /// Reshapes this tensor into a different shape. The new shape must be coherent
    /// with the number of values contained by the current one.
    pub fn reshape(&self) {}

    /// Returns a flattened vector with all the tensor values
    pub fn flatten(&self) {}

    /// Runs `predicate` on every value of the tensor, creating a new tensor with
    /// values obtained from the `predicate` returned ones
    pub fn map<F>(&self, _predicate: F) where F: Fn(TData) -> TData {}
}
