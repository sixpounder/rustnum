use crate::{Shape, generators};
use crate::types::TData;

type TensorValueGenerator<T> = dyn Fn(&Shape) -> T;

pub struct TensorIter {
    coords: Shape,
    value: TData
}

pub enum TensorComponentType {
    Value,
    Axis
}

pub trait TensorComponent {}

fn make_tensor<T: Default + Copy>(shape: Shape, generator: &TensorValueGenerator<T>) -> Tensor<T> {
    let shape_card = shape.mul();
    let mut values: Vec<T> = Vec::with_capacity(shape_card);
    
    // `set_len` would be faster, but unsafe. Consider using it for better performance?
    values.resize(shape_card, T::default());

    for i in shape.iter() {
        values[pos_for(&shape, i.coords)] = generator(i.coords);
    }

    let out_tensor: Tensor<T> = Tensor {
        shape,
        values,
    };

    out_tensor
}

#[derive(Debug)]
pub struct Tensor<T: std::default::Default> {
    values: Vec<T>,
    shape: Shape,
}

impl<T: std::default::Default> TensorComponent for Tensor<T> {}

impl<T: Default + Copy> Tensor<T> {
    pub fn new(shape: Shape, generator: Option<&TensorValueGenerator<T>>) -> Self {
        make_tensor(shape, generator.unwrap_or(&|_| T::default() ))
    }

    pub fn at(&self, coords: Shape) -> Option<&T> {
        let predicted_value = self.values.get(self.index_for_coords(&coords));
        predicted_value
    }

    /// Gets the shape of this tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Reshapes this tensor into a different shape. The new shape must be coherent
    /// with the number of values contained by the current one.
    pub fn reshape(&mut self, t_shape: &Shape) {
        if self.shape.equiv(t_shape) {
            self.shape = t_shape.clone();
        }
    }

    /// Returns a flattened tensor with all the tensor values copied inside it.
    /// This is equivalent to reshaping the tensor to a single dimension equal to the
    /// multiplication of all the axis and cloning it
    pub fn to_flat(&self) -> Tensor<T> {
        Tensor {
            shape: Shape::from(vec![self.shape.mul()]),
            values: self.values.clone(),
        }
    }

    /// Returns a flattened vector with all the tensor values copied inside it
    pub fn to_vec(&self) -> Vec<T> {
        self.values.clone()
    }

    /// Internal utility function for estabilishin a flattened index to assign
    /// coords to
    fn index_for_coords(&self, coords: &Shape) -> usize {
        pos_for(&self.shape, coords)
    }
}

fn pos_for(shape: &Shape, coords: &Shape) -> usize {
    let mut idx = 0;
    for axis in coords.iter_axis() {
        idx += shape.axis_cardinality(*axis).unwrap_or(&0) * axis;
    }

    idx
}

// impl Iterator for Tensor {
//     type Item = TensorIter;
//     fn next(&mut self) -> std::option::Option<<Self as std::iter::Iterator>::Item> {
//         let mut current_tensor = self;
//         let mut current_coords = vec![];
//         let mut axis_index = 0;
//         let mut axis = current_tensor.sub_axis.as_ref().unwrap_or(vec![]);
//         while axis.len() > 0 {
//             if i == current_tensor.shape().len() - 1 {
//                 return Some(TensorIter {
//                     coords: Shape::from(current_coords),
//                     value: current_tensor.values[i]
//                 });
//             } else {
//                 axis = current_tensor.sub_axis.as_ref().unwrap_or(vec![]);
//                 i += 1;
//             }
//         }
//         None
//     }
// }

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn tensor_new() {
        let t = Tensor::<f64>::new(shape!(2, 4, 3), Some(&|shape| { shape.mul() as f64 }));
        assert_eq!(t.shape(), &shape!(2, 4, 3));
    }

    #[test]
    fn tensor_at() {
        let t = Tensor::<f64>::new(shape!(2, 4, 3), Some(&|shape| { shape.mul() as f64 }));
        let test_value = t.at(shape!(0, 1, 1));
        assert_eq!(test_value, Some(&24.0));
    }
}
