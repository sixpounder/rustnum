use std::ops::{Index, IndexMut};

use crate::{Coord, Shape};

type TensorValueGenerator<T> = dyn Fn(&Shape, u64) -> T;

pub trait TensorComponent {}

fn pos_for(shape: &Shape, coords: &Shape) -> usize {
    let mut idx = 0;
    for axis in coords.iter_axis() {
        idx += shape.axis_cardinality(*axis).unwrap_or(&0) * axis;
    }

    idx
}

fn make_tensor<T: Default + Copy>(shape: Shape, generator: &TensorValueGenerator<T>) -> Tensor<T> {
    let shape_card = shape.mul();
    let mut values: Vec<T> = Vec::with_capacity(shape_card);
    
    // `set_len` would be faster, but unsafe. Consider using it for better performance?
    values.resize(shape_card, T::default());
    let mut counter = 0;
    for i in shape.iter() {
        values[pos_for(&shape, &i)] = generator(&i, counter);
        counter += 1;
    }

    let out_tensor: Tensor<T> = Tensor {
        shape,
        values,
    };

    out_tensor
}

#[derive(Debug)]
pub struct Tensor<T> {
    values: Vec<T>,
    shape: Shape,
}

impl<T: std::default::Default> TensorComponent for Tensor<T> {}

impl<T> Tensor<T> {
    /// Internal utility function for estabilishin a flattened index to assign
    /// coords to
    fn index_for_coords(&self, coords: &Shape) -> Option<usize> {
        let idx = pos_for(&self.shape, coords);
        if self.values.len() > idx {
            Some(idx)
        } else {
            None
        }
    }

    pub fn at(&self, coords: Shape) -> Option<&T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => {
                self.values.get(index)
            },
            None => None
        }
    }

    pub fn at_ref(&self, coords: &Shape) -> Option<&T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => {
                self.values.get(index)
            },
            None => None
        }
    }

    pub fn at_mut(&mut self, coords: Shape) -> Option<&mut T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => {
                self.values.get_mut(index)
            },
            None => None
        }
    }

    pub fn set(&mut self, coords: Coord, value: T) -> Result<(), ()> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => {
                self.values[index] = value;
                Ok(())
            },
            None => Err(())
        }
    }

    pub fn iter(&self) -> TensorIterator<T> {
        TensorIterator {
            tensor: &self,
            next_coord: None,
            started: false
        }
    }

    // pub fn sum(&self, other: Tensor<T>) -> Result<Tensor<T>> {
    //     if self.shape.equiv(other.shape()) {

    //     }
    // }
}

impl<T: Default + Copy> Tensor<T> {

    /// Creates a new tensor
    pub fn new(shape: Shape, generator: Option<&TensorValueGenerator<T>>) -> Self {
        make_tensor(shape, generator.unwrap_or(&|_, _| T::default() ))
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
}

impl<T> Index<Coord> for Tensor<T> {
    type Output = T;

    fn index(&self, index: Coord) -> &Self::Output {
        self.at(index).unwrap()
    }
}

impl<T> IndexMut<Coord> for Tensor<T> {
    fn index_mut(&mut self, index: Coord) -> &mut Self::Output {
        self.at_mut(index).unwrap()
    }
}

// impl<T> Add for Tensor<T> {
//     type Output = Tensor<T>;

//     fn add(self, rhs: Self) -> Self::Output {
//         self.sum(rhs)
//     }
// }

#[derive(Debug, Clone)]
pub struct TensorIter<'a, T> {
    coords: Shape,
    value: &'a T
}

pub struct TensorIterator<'a, T> {
    started: bool,
    tensor: &'a Tensor<T>,
    next_coord: Option<Shape>,
}

impl<'a, T> TensorIterator<'a, T> {
    pub fn has_next(&self) -> bool {
        self.started && self.next_coord.is_some()
    }

    fn step_forward(&mut self) {
        if !self.started {
            self.started = true;
            self.next_coord = Some(self.tensor.shape.clone());
        } else {
            let current_coord = self.next_coord.as_ref().unwrap();
            self.next_coord = Shape::next_coord(&self.tensor.shape, &current_coord);
        }
    }
    
}

impl<'a, T> Iterator for TensorIterator<'a, T> {
    type Item = TensorIter<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.step_forward();
        if self.has_next() {
            let next_iter: TensorIter<'a, T> = TensorIter {
                coords: self.next_coord.as_ref().unwrap().clone(),
                value: self.tensor.at_ref(self.next_coord.as_ref().unwrap()).unwrap()
            };
    
            Some(next_iter)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn tensor_new() {
        let t = Tensor::<f64>::new(shape!(2, 4, 3), Some(&|shape, _| { shape.mul() as f64 }));
        assert_eq!(t.shape(), &shape!(2, 4, 3));
    }

    #[test]
    fn tensor_at() {
        let generator: &TensorValueGenerator<f64> = &|coord, _| { coord.mul() as f64 };
        let t = Tensor::<f64>::new(shape!(2, 4, 3), Some(generator));
        assert_eq!(t.shape(), &shape!(2, 4, 3));
        let test_value = t.at(shape!(1, 2, 2));
        assert_eq!(test_value, Some(&4.0));
    }
}
