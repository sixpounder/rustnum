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
    let mut values = vec![];
    let mut sub_axis = vec![];
    if shape.len() > 1 {
        for axis in 0..shape[0] {
            sub_axis.push(Box::new(makeTensor(Shape::from(vec![*shape.last().unwrap()]), generator, at_coords)));
        }
    } else {
        values.push(generator(at_coords.unwrap()));
    }

    let out_tensor = Tensor {
        shape,
        values,
        sub_axis: Some(sub_axis),
    };

    out_tensor
}

#[derive(Debug)]
pub struct Tensor {
    values: Vec<TData>,
    sub_axis: Option<Vec<Box<Tensor>>>,
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
            values.push(spawn_value(&Shape::from(vec![i])));
        }

        Tensor {
            shape,
            values,
            sub_axis: Some(vec![]),
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
    pub fn flatten(&self) -> Tensor {
        Tensor {
            shape: Shape::from(vec![self.shape.mul()]),
            values: vec![],
            sub_axis: None
        }
    }

    /// Runs `predicate` on every value of the tensor, creating a new tensor with
    /// values obtained from the `predicate` returned ones
    pub fn map<F>(&self, _predicate: F) where F: Fn(Shape, TData) -> TData {}
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn tensor_new() {
        let t = Tensor::new(shape!(2, 4), None);
        assert_eq!(t.shape(), &shape!(2, 4));
    }
}
