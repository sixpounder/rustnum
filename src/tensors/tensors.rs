use std::{mem::zeroed, ops::{Add, Index, IndexMut, Mul}};

use crate::{ops::Dot, Coord, CoordIterator, Shape};

type TensorValueGenerator<T> = dyn Fn(&Coord, u64) -> T;

/// Enumeration for common errors on tensors
#[derive(Debug, PartialEq)]
pub enum TensorError {
    Init,
    Set,
    NotSupported,
    /// Usually thrown when an operation on a tensor
    /// involving a set of coordinates not contained by the tensor was attempted
    NoCoordinate,
}

fn pos_for(shape: &Shape, coords: &Coord) -> usize {
    let mut idx = 0;
    let mut axis_index: usize = 0;
    for axis in coords.iter_axis() {
        idx += shape.axis_cardinality(axis_index).unwrap_or(&0) * axis;
        axis_index += 1;
    }

    idx
}

fn make_tensor<T>(shape: &Shape, generator: &TensorValueGenerator<T>) -> Tensor<T> {
    let shape_card = shape.mul();
    let mut values: Vec<T> = Vec::with_capacity(shape_card);

    // `set_len` unsafe, but fast
    unsafe {
        values.set_len(shape_card);
    }
    let mut counter = 0;
    for i in shape.iter() {
        let position = pos_for(&shape, &i);
        values[position] = generator(&i, counter);
        counter += 1;
    }

    let out_tensor: Tensor<T> = Tensor {
        shape: shape.clone(),
        values,
    };

    out_tensor
}

/// A tensor is simply a multidimensional array containing items of some kind. It is the struct
/// returned by many of the distribution generators in this crate.
///
/// ## Create a tensor, read and write values
/// ```
/// # use rustnum::{Tensor, shape, Shape, Coord, coord};
/// let generator = &|coord: &Coord, counter: u64| {
///     // For example, make every number equal to
///     // the cardinality of the coordinate plus the counter
///     coord.cardinality() as f64 + counter as f64
/// };
///
/// let mut tensor: Tensor<f64> = Tensor::new(
///     shape!(3, 4, 10),
///     Some(generator)
/// );
///
///
/// // Get values
/// tensor.at(coord!(0, 0, 1));
/// // or
/// tensor[coord!(0, 0, 2)];
///
/// // Set values
/// tensor[coord!(0, 1, 2)] = 0.5;
/// // or
/// tensor.set(&coord!(0, 1, 2), 0.5);
/// ```
#[derive(Debug)]
pub struct Tensor<T> {
    values: Vec<T>,
    shape: Shape,
}

impl<T> Tensor<T> {
    /// Same as `new`, but Creates an empty tensor with uninitialized memory.
    /// This is significantly faster than `new`
    /// but it leaves the tensor elements into uninitialized memory state.
    /// **It is up to the user to guarantee initialization of each element of the tensor**,
    /// or each access will resolve into undefined behavior.
    ///
    /// ```
    /// # use rustnum::{Tensor, shape, coord, Shape, Coord};
    /// let empty: Tensor<f64> = Tensor::new_uninit(shape!(4, 7));
    /// assert!(empty.at(coord!(0, 0)).is_some());
    /// ```
    pub fn new_uninit(shape: Shape) -> Tensor<T> {
        make_tensor(&shape, &|_, _| unsafe {
            // std::mem::MaybeUninit::uninit().assume_init()
            std::mem::MaybeUninit::zeroed().assume_init()
        })
    }

    /// Gets the shape of this tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Gets the shape of this tensor as mutable
    pub fn shape_mut(&mut self) -> &mut Shape {
        &mut self.shape
    }

    /// Internal utility function for estabilishin a flattened index to assign
    /// coords to
    fn index_for_coords(&self, coords: &Coord) -> Option<usize> {
        let idx = pos_for(&self.shape, coords);
        if self.values.len() > idx {
            Some(idx)
        } else {
            None
        }
    }

    /// Gets the value at coordinates `coord`, if any
    pub fn at(&self, coords: Coord) -> Option<&T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => self.values.get(index),
            None => None,
        }
    }

    /// Gets the value at coordinates `coord`, if any
    pub fn at_ref(&self, coords: &Coord) -> Option<&T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => self.values.get(index),
            None => None,
        }
    }

    /// Gets the value at coordinates `coord`, if any. The returned value is mutable.
    pub fn at_mut(&mut self, coords: Coord) -> Option<&mut T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => self.values.get_mut(index),
            None => None,
        }
    }

    /// Gets the value at coordinates `&coord`, if any. The returned value is mutable.
    pub fn at_ref_mut(&mut self, coords: &Coord) -> Option<&mut T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => self.values.get_mut(index),
            None => None,
        }
    }

    /// The total number of items in this tensor
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Sets the value at coordinates `coord`. If `coord` is not contained by this tensor it returns an error
    /// ```
    /// # use rustnum::{Tensor, shape, coord, Shape, Coord};
    /// let mut t = Tensor::<u8>::new_uninit(shape!(4, 5, 6));
    /// assert_eq!(t[coord!(0, 0, 0)], 0);
    /// assert_ne!(t.set(&coord!(0, 0, 0), 8), Err(TensorError::Set));
    /// assert_eq!(t[coord!(0, 0, 0)], 8);
    ///
    /// // Trying to set a value on a non existing coordinate will yield an error
    /// assert_eq!(t.set(&coord!(4, 1, 10), 10), Err(TensorError::NoCoordinate));
    /// ```
    pub fn set(&mut self, coords: &Coord, value: T) -> Result<(), TensorError> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => {
                self.values[index] = value;
                Ok(())
            }
            None => Err(TensorError::NoCoordinate),
        }
    }

    /// Returns an iterator over this tensor
    pub fn iter(&self) -> TensorIterator<T> {
        TensorIterator::new(self)
    }
}

impl<T: Copy> Tensor<T> {
    /// Reshapes this tensor into a different shape. The new shape must be coherent
    /// with the number of values contained by the current one.
    pub fn reshape(&mut self, t_shape: Shape) {
        if self.shape.equiv(&t_shape) {
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

    pub fn iter_mut(&mut self) -> TensorIteratorMut<T> {
        TensorIteratorMut::new(self)
    }
}

impl<T: Default> Tensor<T> {
    /// Creates a new tensor with a value generator
    /// # Example
    /// ```
    /// # use rustnum::{Tensor, shape, Shape, Coord};
    /// let generator = &|coord: &Coord, counter: u64| {
    ///     // For example, make every number equal to
    ///     // the cardinality of the coordinate plus the counter
    ///     coord.cardinality() as f64 + counter as f64
    /// };
    ///
    /// let tensor: Tensor<f64> = Tensor::new(
    ///     shape!(3, 4, 10),
    ///     Some(generator)
    /// );
    /// ```
    ///
    pub fn new(shape: Shape, generator: Option<&TensorValueGenerator<T>>) -> Self {
        make_tensor(&shape, generator.unwrap_or(&|_, _| T::default()))
    }
}

impl<T> Tensor<T>
where
    T: Copy + std::ops::Mul<Output = T>,
{
    /// Scales up every item in this tensor by `scalar`
    pub fn scale(&mut self, scalar: T) {
        for item in self.values.iter_mut() {
            *item = *item * scalar;
        }
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

impl<T: Clone> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.clone(),
        }
    }
}

// Implements scalar multiplication
impl<T: Copy + std::ops::Mul<Output = T>> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut out_tensor: Tensor<T> = self.clone();
        out_tensor.scale(rhs);

        // out_tensor
        for item in self.iter() {
            out_tensor
                .set(&item.coords, (*item.value * rhs).into())
                .unwrap();
        }

        out_tensor
    }
}

// Implements scalar sum
impl<T: Copy + std::ops::Add<Output = T>> Add<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        let mut out_tensor = self.clone();
        for item in out_tensor.iter_mut() {
            *item.value = *item.value + rhs;
        }

        out_tensor
    }
}

// impl<T> Dot<Tensor<T>> for Tensor<T> where T: std::ops::Mul {
//     type Output = Result<Tensor<T>, TensorError>;

//     fn dot(&self, b: Tensor<T>) -> Self::Output {
//         let a_shape = self.shape();
//         let b_shape = b.shape();
//         if a_shape.len() > 2 || b.shape().len() > 2 {
//             Err(TensorError::NotSupported)
//         } else {
//             let target_shape: Shape;
//             if a_shape[1] == b_shape[0] {
//                 target_shape = shape!(a_shape[0], b_shape[1]);
//             } else {
//                 return Err(TensorError::NotSupported);
//             }

//             let out_tensor = Tensor::<T>::new_uninit(target_shape);

//             for row in self.iter() {
//                 let r_sum = 0.0;
//                 for col in b.iter() {
//                     r_sum = r_sum + (*row.value * *col.value);
//                 }
//             }

//             Ok(out_tensor)
//         }
//     }
// }

/// A single component of a tensor
#[derive(Debug)]
pub struct TensorComponent<'a, T> {
    pub coords: Coord,
    pub value: &'a T,
    pub tensor: &'a Tensor<T>,
}

impl<T: std::cmp::PartialEq> PartialEq for TensorComponent<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        *self.value == *other.value
    }
}

impl<T: PartialEq> Eq for TensorComponent<'_, T> {}

impl<T: Ord> PartialOrd for TensorComponent<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((*self.value).cmp(other.value))
    }
}

impl<T: PartialEq + Ord> Ord for TensorComponent<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if *self.value < *other.value {
            std::cmp::Ordering::Less
        } else if *self.value > *other.value {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

/// Implements an iteration over a tensor
pub struct TensorIterator<'a, T> {
    tensor: &'a Tensor<T>,
    coord_iter: CoordIterator<'a>,
}

impl<'a, T> TensorIterator<'a, T> {
    pub fn new(tensor: &'a Tensor<T>) -> Self {
        Self {
            tensor,
            coord_iter: CoordIterator::new(tensor.shape()),
        }
    }
}

impl<'a, T> Iterator for TensorIterator<'a, T> {
    type Item = TensorComponent<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.coord_iter.next() {
            Some(item) => {
                let value = self.tensor.at_ref(&item).unwrap();
                Some(TensorComponent {
                    coords: item,
                    value,
                    tensor: self.tensor,
                })
            }
            None => None,
        }
    }
}

/// A single mutable component of a tensor
#[derive(Debug)]
pub struct TensorComponentMut<'a, T> {
    pub coords: Coord,
    pub value: &'a mut T,
}

/// Implements a mutable iterator over a tensor
pub struct TensorIteratorMut<'a, T> {
    tensor: *mut Tensor<T>,
    coord_iter: CoordIterator<'a>,
}

impl<'a, T> TensorIteratorMut<'a, T> {
    pub fn new(tensor: &'a mut Tensor<T>) -> Self {
        let t: *mut Tensor<T> = &mut *tensor;
        let t_shape = tensor.shape();
        let coord_iter = CoordIterator::new(t_shape);
        Self {
            tensor: t,
            coord_iter,
        }
    }
}

impl<'a, T: 'a> Iterator for TensorIteratorMut<'a, T> {
    type Item = TensorComponentMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.coord_iter.next() {
            Some(item) => unsafe {
                let tensor_ptr: *mut Tensor<T> = self.tensor;
                if tensor_ptr.is_null() {
                    None
                } else {
                    let value = (*tensor_ptr).at_ref_mut(&item).unwrap();
                    Some(TensorComponentMut {
                        coords: item,
                        value,
                    })
                }
            },
            None => None,
        }
    }
}

impl<T> Drop for TensorIteratorMut<'_, T> {
    fn drop(&mut self) {
        if !self.tensor.is_null() {
            drop(self.tensor);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn new() {
        let t = Tensor::<f64>::new(
            shape!(2, 4, 3),
            Some(&|coord, _| coord.cardinality() as f64),
        );
        assert_eq!(t.shape(), &shape!(2, 4, 3));
    }

    #[test]
    fn at() {
        let generator: &TensorValueGenerator<f64> = &|coord, _| coord.cardinality() as f64;
        let t = Tensor::<f64>::new(shape!(2, 4, 3), Some(generator));
        assert_eq!(t.shape(), &shape!(2, 4, 3));
        let test_value = t.at(coord!(1, 2, 2));
        assert_eq!(test_value, Some(&4.0));
    }

    #[test]
    fn iter_mut() {
        let generator: &TensorValueGenerator<f64> = &|coord, _| coord.cardinality() as f64;
        let mut t = Tensor::<f64>::new(shape!(2, 4, 3), Some(generator));
        let c = t.iter_mut().count();
        assert_eq!(c, 24);
    }

    #[test]
    fn scalar_sum() {
        let generator: &TensorValueGenerator<f64> = &|coord, _| coord.cardinality() as f64;
        let mut t = Tensor::<f64>::new(shape!(2, 4, 3), Some(generator));
        t = t + 2.0;
        assert_eq!(t.at(coord!(1, 2, 2)), Some(&6.0));
    }

    #[test]
    fn scalar_mul() {
        let generator: &TensorValueGenerator<f64> = &|coord, _| coord.cardinality() as f64;
        let mut t = Tensor::<f64>::new(shape!(2, 4, 3), Some(generator));
        t = t * 2.0;
        assert_eq!(t.at(coord!(1, 2, 2)), Some(&8.0));
    }

    #[test]
    fn ordering() {
        let generator: &TensorValueGenerator<u64> = &|_, i| i;
        let t = Tensor::<u64>::new(shape!(2, 4, 3), Some(generator));

        let min_component = t.iter().min().unwrap();
        assert_eq!(min_component.value, &0);
        assert_eq!(min_component.coords, coord!(0, 0, 0));

        let max_component = t.iter().max().unwrap();
        assert_eq!(max_component.value, &23);
        assert_eq!(max_component.coords, coord!(1, 3, 2));
    }

    #[test]
    fn uninit() {
        let mut t = Tensor::<u8>::new_uninit(shape!(4, 5, 6));
        assert_eq!(t[coord!(0, 0, 0)], 0);
        assert_ne!(t.set(&coord!(0, 0, 0), 8), Err(TensorError::Set));
        assert_eq!(t[coord!(0, 0, 0)], 8);
        assert_eq!(t.set(&coord!(4, 1, 10), 10), Err(TensorError::NoCoordinate));
    }

    #[test]
    fn uninit_at_no_panic() {
        let t = Tensor::<u8>::new_uninit(shape!(4, 5, 6));
        assert!(t.at(coord!(0, 1, 3)).is_some());
    }
}
