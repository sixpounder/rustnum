use crate::Shape;
use crate::types::TData;
use std::str::FromStr;

type TensorValueGenerator = dyn Fn(&Shape) -> TData;

pub enum TensorComponentType {
    Value,
    Axis
}

pub trait TensorComponent {}



fn makeTensor(shape: Shape, generator: &TensorValueGenerator, at_coords: Option<&Shape>) -> Tensor {
    let mut out_tensor = Tensor {
        shape: shape,
        values: vec![],
        sub_axis: vec![],
    };

    if out_tensor.shape.len() > 1 {
        for axis in 0..out_tensor.shape[0] {
            out_tensor.sub_axis.push(Box::new(makeTensor(vec![*out_tensor.shape.last().unwrap()], generator, at_coords)));
        }
    } else {
        out_tensor.values.push(generator(at_coords.unwrap()));
    }

    out_tensor
}

#[derive(Debug)]
pub struct Tensor {
    values: Vec<TData>,
    sub_axis: Vec<Box<Tensor>>,
    shape: Shape,
}

impl TensorComponent for Tensor {}

impl Tensor {
    pub fn new(shape: Shape, generator: Option<&TensorValueGenerator>) -> Self {
        // makeTensor(shape, generator.unwrap_or(&|_| 0.0 ), None)
        let mut n_values = 1;
        shape.iter().for_each(|i| { n_values *= i } );

        let mut values = vec![];
        let spawn_value = generator.unwrap_or(&|coords| f64::from_str(&coords[0].to_string()).unwrap());

        for i in 0..n_values {
            values.push(spawn_value(&vec![i]));
        }

        Tensor {
            shape,
            values,
            sub_axis: vec![],
        }
    }

    pub fn at(&self, coords: Shape) {

    }

    /// Gets the shape of this tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Reshapes this tensor into a different shape. The new shape must be coherent
    /// with the number of values contained by the current one.
    pub fn reshape(&self, t_shape: Shape) {}

    /// Returns a flattened vector with all the tensor values
    pub fn flatten(&self) {}

    /// Runs `predicate` on every value of the tensor, creating a new tensor with
    /// values obtained from the `predicate` returned ones
    pub fn map<F>(&self, _predicate: F) where F: Fn(TData) -> TData {}
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn tensor_new() {
        let t = Tensor::new(vec![2, 4], None);
        assert_eq!(t.shape(), vec![2usize, 4usize]);
    }
}
