use crate::Shape;
use crate::types::TData;

type TensorValueGenerator = dyn Fn(&Shape) -> TData;

pub enum TensorComponentType {
    Value,
    Axis
}

pub trait TensorComponent {}

fn make_tensor(shape: Shape, generator: &TensorValueGenerator, at_coords: Option<&Shape>) -> Tensor {
    let mut values = vec![];
    let mut sub_axis = vec![];
    if shape.len() > 1 {
        for axis in 0..shape[0] {
            let next_coord = match at_coords {
                Some(c) => {
                    let mut new_shape = c.clone();
                    new_shape.append(axis);
                    new_shape
                }
                None => {
                    Shape::new(vec![axis])
                }
            };

            sub_axis.push(Box::new(make_tensor(shape.tail(), generator, Some(&next_coord))));
        }
    } else {
        match at_coords {
            Some(coords) => {
                values.push(generator(&coords));
            },
            None => ()
        }
    }

    let out_tensor = Tensor {
        shape,
        values,
        sub_axis: if sub_axis.len() > 0 { Some(sub_axis) } else { None },
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
        make_tensor(shape, generator.unwrap_or(&|_| 0.0 ), None)
    }

    pub fn at(&self, _coords: Shape) {

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
