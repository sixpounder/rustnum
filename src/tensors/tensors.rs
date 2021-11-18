use crate::{
    coord,
    ops::{Dot, Stats},
    shape, Coord, CoordIterator, Set, Shape,
};
use num_traits::{Float, Num};
use std::{
    any::Any,
    fmt::{Display, Formatter},
    ops::{Add, Index, IndexMut, Mul},
};

/// Enumeration for common errors on tensors
#[derive(Debug, PartialEq)]
pub enum TensorError {
    /// Tensor initialization error
    Init,

    /// An error occurred while setting a value on a tensor
    Set,

    /// This operation is not supported on tensors
    NotSupported,

    /// Usually thrown when an operation on a tensor
    /// involving a set of coordinates not contained by the tensor was attempted
    NoCoordinate,

    /// Some thread-related error
    Thread,
}

impl<T> From<std::result::Result<T, Box<dyn Any + Send + 'static>>> for TensorError {
    fn from(_: std::result::Result<T, Box<dyn Any + Send + 'static>>) -> Self {
        Self::Thread
    }
}

#[inline]
fn pos_for(shape: &Shape, coords: &Coord) -> usize {
    let mut idx = 0;
    let mut axis_index: usize = 0;
    for axis in coords.iter_axis() {
        idx += shape.axis_cardinality(axis_index).unwrap_or(&0) * axis;
        axis_index += 1;
    }

    idx
}

pub trait TensorLike<T> {
    fn shape(&self) -> Shape;
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    fn at(&self, coords: Coord) -> Option<&T>;
    fn set(&mut self, coords: Coord, value: T) -> Result<(), TensorError>;
    fn first(&self) -> Option<&T> {
        self.at(Coord::zeroes(self.shape().ndim()))
    }
    fn is_scalar(&self) -> bool {
        self.shape().len() == 0
    }
}

impl<T> TensorLike<T> for T
where
    T: Num,
{
    fn shape(&self) -> Shape {
        shape!()
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn at(&self, _coords: Coord) -> Option<&T> {
        Some(self)
    }

    fn set(&mut self, _coords: Coord, _value: T) -> Result<(), TensorError> {
        Ok(())
    }
}

impl<T> TensorLike<T> for Vec<T> {
    fn shape(&self) -> Shape {
        shape!(self.len())
    }

    fn at(&self, coords: Coord) -> Option<&T> {
        self.get(coords[0])
    }

    fn set(&mut self, coords: Coord, value: T) -> Result<(), TensorError> {
        self[coords.cardinality()] = value;
        Ok(())
    }

    fn rank(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().cardinality()
    }
}

/// A tensor is simply a multidimensional array containing items of some kind. It is the struct
/// returned by many of the distribution generators in this crate.
///
/// ## Create a tensor, read and write values
/// ```
/// # use rustnum::{Tensor, shape, Shape, Coord, coord, TensorLike};
/// let generator = |coord: Coord, counter: usize| {
///     // For example, make every number equal to
///     // the cardinality of the coordinate plus the counter
///     coord.cardinality() as f64 + counter as f64
/// };
///
/// let mut tensor: Tensor<f64> = Tensor::new(
///     shape!(3, 4, 10),
///     generator
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
/// tensor.set(coord!(0, 1, 2), 0.5);
/// ```
#[derive(Debug)]
pub struct Tensor<T> {
    values: Vec<T>,
    shape: Shape,
}

impl<T> TensorLike<T> for Tensor<T> {
    /// Gets the shape of this tensor
    #[inline]
    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    // #[inline]
    // fn values(&self) -> Vec<&T> {
    //     self.values.iter().collect()
    // }

    /// Gets the value at coordinates `coord`, if any
    #[inline]
    fn at(&self, coords: Coord) -> Option<&T> {
        if self.shape().is_scalar() {
            Some(&self.values[0])
        } else {
            let idx = self.index_for_coords(&coords);
            match idx {
                Some(index) => self.values.get(index),
                None => None,
            }
        }
    }

    /// Sets the value at coordinates `coord`. If `coord` is not contained by this tensor it returns an error
    /// ```
    /// # use rustnum::{Tensor, shape, coord, Shape, Coord, TensorError, TensorLike};
    /// let mut t = Tensor::<u8>::new_uninit(shape!(4, 5, 6));
    /// assert_eq!(t[coord!(0, 0, 0)], 0);
    /// assert_ne!(t.set(coord!(0, 0, 0), 8), Err(TensorError::Set));
    /// assert_eq!(t[coord!(0, 0, 0)], 8);
    ///
    /// // Trying to set a value on a non existing coordinate will yield an error
    /// assert_eq!(t.set(coord!(4, 1, 10), 10), Err(TensorError::NoCoordinate));
    /// ```
    #[inline]
    fn set(&mut self, coords: Coord, value: T) -> Result<(), TensorError> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => {
                self.values[index] = value;
                Ok(())
            }
            None => Err(TensorError::NoCoordinate),
        }
    }

    fn rank(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().cardinality()
    }
}

impl<T> Tensor<T> {
    fn make_tensor<G>(shape: Shape, generator: G) -> Tensor<T>
    where
        G: Fn(Coord, usize) -> T,
    {
        let shape_card = shape.cardinality();
        let mut values: Vec<T> = Vec::with_capacity(shape_card);

        // `set_len` unsafe, but fast
        unsafe {
            values.set_len(shape_card);
        }
        let mut counter: usize = 0;
        for i in shape.iter() {
            let position = pos_for(&shape, &i);
            values[position] = generator(i, counter);
            counter += 1;
        }

        let out_tensor: Tensor<T> = Tensor {
            shape: shape.clone(),
            values,
        };

        out_tensor
    }

    pub fn scalar(value: T) -> Self {
        Self {
            shape: shape!(),
            values: vec![value],
        }
    }
    /// Same as `new`, but Creates an empty tensor with uninitialized memory.
    /// This is significantly faster than `new`
    /// but it leaves the tensor elements into uninitialized memory state.
    /// **It is up to the user to guarantee initialization of each element of the tensor**,
    /// or each access will resolve into undefined behavior.
    ///
    /// ```
    /// # use rustnum::{Tensor, shape, coord, Shape, Coord, TensorLike};
    /// let empty: Tensor<f64> = Tensor::new_uninit(shape!(4, 7));
    /// assert!(empty.at(coord!(0, 0)).is_some());
    /// ```
    pub fn new_uninit(shape: Shape) -> Tensor<T> {
        Self::make_tensor(shape, |_, _| unsafe {
            std::mem::MaybeUninit::zeroed().assume_init()
        })
    }

    /// Creates a new tensor with a value generator
    /// # Example
    /// ```
    /// # use rustnum::{Tensor, shape, Shape, Coord, TensorLike};
    ///
    /// let tensor: Tensor<f64> = Tensor::new(
    ///     shape!(3, 4, 10),
    ///     |coord: Coord, counter| {
    ///         // For example, make every number equal to
    ///         // the cardinality of the coordinate plus the counter
    ///         coord.cardinality() as f64 + counter as f64
    ///     }
    /// );
    /// ```
    ///
    pub fn new<G>(shape: Shape, generator: G) -> Self
    where
        G: Fn(Coord, usize) -> T,
    {
        Self::make_tensor(shape, generator)
    }

    pub fn from_vec_with_shape(values: Vec<T>, shape: Shape) -> Tensor<T> {
        let mut new_tensor = Self::from(values);
        new_tensor.reshape(shape);

        new_tensor
    }

    /// Gets the shape of this tensor
    pub fn shape_ref(&self) -> &Shape {
        &self.shape
    }

    /// Internal utility function for estabilishin a flattened index to assign
    /// coords to
    #[inline]
    fn index_for_coords(&self, coords: &Coord) -> Option<usize> {
        let idx = pos_for(&self.shape, coords);
        if self.values.len() > idx {
            Some(idx)
        } else {
            None
        }
    }

    /// Gets the value at coordinates `coord`, if any
    #[inline]
    pub fn at_ref(&self, coords: &Coord) -> Option<&T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => self.values.get(index),
            None => None,
        }
    }

    /// Gets the value at coordinates `coord`, if any. The returned reference is mutable.
    #[inline]
    pub fn at_mut(&mut self, coords: Coord) -> Option<&mut T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => self.values.get_mut(index),
            None => None,
        }
    }

    /// Gets the value at coordinates `coord`, if any, using a reference to coords instead of a
    /// moved value. The returned reference is mutable.
    #[inline]
    pub fn at_ref_mut(&mut self, coords: &Coord) -> Option<&mut T> {
        let idx = self.index_for_coords(&coords);
        match idx {
            Some(index) => self.values.get_mut(index),
            None => None,
        }
    }

    /// Returns an iterator over this tensor. The iteration order is defined by its shape, meaning it will iter by axis
    /// priority. For example, a tensor like this:
    /// ```ignore
    /// # use crate::tensor;
    /// let t1 = tensor!((2, 2, 3), [12, 23, 32, 0, 2, 23, 12, 23, 32, 0, 2, 23]);
    /// ```
    /// will be iterated like this order in coordinates:
    /// * (0, 0, 0)
    /// * (0, 0, 1)
    /// * (0, 0, 2)
    /// * (0, 1, 0)
    /// * (0, 1, 2)
    /// * ...
    pub fn iter(&self) -> TensorIterator<T> {
        TensorIterator::new(self)
    }

    pub fn enumerate(&self) -> TensorEnumerator<T> {
        TensorEnumerator::new(self)
    }

    /// Same as `iter`, but yelding mutable refs to tensor components
    pub fn iter_mut(&mut self) -> TensorIteratorMut<T> {
        TensorIteratorMut::new(self)
    }

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
    pub fn to_flat(self) -> Tensor<T> {
        Tensor {
            shape: Shape::from(vec![self.shape.cardinality()]),
            values: self.values,
        }
    }

    /// Returns a flattened vector with all the tensor values copied inside it
    pub fn to_vec(self) -> Vec<T> {
        self.values
    }
}

impl<T: Copy> Tensor<T> {
    /// Casts a tensor with some base type to another, given that the old type is
    /// transformable into the new one
    pub fn cast<O>(&self) -> Tensor<O>
    where
        T: Into<O>,
    {
        Tensor::new(self.shape(), |coord, _counter| self[coord].into())
    }
}

impl<T: Default> Tensor<T> {
    /// Creates a new tensor with a value generator
    /// # Example
    /// ```
    /// # use rustnum::{Tensor, shape, Shape, Coord, TensorLike};
    /// let generator = |coord: Coord, counter: usize| {
    ///     // For example, make every number equal to
    ///     // the cardinality of the coordinate plus the counter
    ///     coord.cardinality() as f64 + counter as f64
    /// };
    ///
    /// let tensor: Tensor<f64> = Tensor::new(
    ///     shape!(3, 4, 10),
    ///     generator
    /// );
    /// ```
    ///
    pub fn new_with_default<G>(shape: Shape, generator: Option<G>) -> Self
    where
        G: Fn(Coord, usize) -> T,
    {
        match generator {
            Some(f) => Self::make_tensor(shape, f),
            None => Self::make_tensor(shape, |_, _| T::default()),
        }
    }
}

impl<T> Tensor<T>
where
    T: Num + Copy,
{
    /// Scales up every item in this tensor by `scalar`
    pub fn scalar_mul(&mut self, scalar: T) {
        for item in self.values.iter_mut() {
            *item = *item * scalar;
        }
    }
}

impl<T> Tensor<T>
where
    T: Num,
{
    /// Creates a tensor with a given shape and all values set to 1.
    /// "1" is taken to be of the appropriate base type, which can be any type
    /// for which the `Num` trait is implemented
    pub fn ones(shape: Shape) -> Tensor<T> {
        Self::make_tensor(shape, |_, _| T::one())
    }

    /// Creates a tensor with a given shape and all values set to 0.
    /// "0" is taken to be of the appropriate base type, which can be any type
    /// for which the `Num` trait is implemented
    pub fn zeros(shape: Shape) -> Tensor<T> {
        Self::make_tensor(shape, |_, _| T::zero())
    }

    // Outer product between two tensors. Consumes self.
    // pub fn tensor_product<Rhs, Y>(self, other: &Rhs) -> Tensor<T> where Rhs: TensorLike<Y>, Y: Num {
    //     let output_shape = self.shape() * other.shape();
    // }
}

impl<T> From<Vec<T>> for Tensor<T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            shape: values.shape(),
            values,
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

impl<Rhs, T> Mul<Rhs> for Tensor<T>
where
    Rhs: TensorLike<T>,
    T: Copy + Num,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        let mut out_tensor: Tensor<T> = self.clone();
        if rhs.is_scalar() {
            out_tensor.scalar_mul(*rhs.first().unwrap());
        } else {
            for item in self.iter() {
                let coord = item.coords.clone();
                out_tensor
                    .set(item.coords, (*item.value * *rhs.at(coord).unwrap()).into())
                    .unwrap();
            }
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

impl<T> Add<Tensor<T>> for Tensor<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape());
        let mut new_tensor = self.clone();
        self.iter().for_each(|component| {
            let coord = component.coords.clone();
            new_tensor
                .set(
                    component.coords,
                    component.value.clone() + rhs.at_ref(&coord).unwrap().clone(),
                )
                .unwrap();
        });

        new_tensor
    }
}

impl<T> Display for Tensor<T>
where
    T: Display + Copy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut o: String = String::new();
        let mut last_emitted_symbol = "";
        for e in self.enumerate() {
            match e {
                EnumerationPoint::AxisBegin => {
                    o = o.add(
                        format!(
                            "{}[",
                            match last_emitted_symbol {
                                "]" => ", ",
                                "[" => "",
                                _ => "",
                            }
                        )
                        .as_str(),
                    );
                    last_emitted_symbol = "[";
                }
                EnumerationPoint::AxisEnd => {
                    o = o.add(
                        format!(
                            "{}]",
                            match last_emitted_symbol {
                                "t" => " ",
                                _ => "",
                            }
                        )
                        .as_str(),
                    );
                    last_emitted_symbol = "]";
                }
                EnumerationPoint::Terminal(component) => {
                    o = o.add(
                        format!(
                            "{}{}",
                            match last_emitted_symbol {
                                "t" => ", ",
                                "[" => " ",
                                _ => "",
                            },
                            component.value,
                            // match last_emitted_symbol {
                            //     "t" => ", ",
                            //     "[" => " ",
                            //     _ => "",
                            // }
                        )
                        .as_str(),
                    );
                    last_emitted_symbol = "t";
                }
            }
        }

        write!(f, "{}", o.as_str())
    }
}

impl<T: Float> Stats for Tensor<T> {
    type Output = T;

    fn mean(&self) -> Self::Output {
        let mut collector: T = T::zero();
        self.iter().for_each(|component| {
            collector = collector.add(*component.value);
        });

        match T::from(self.size()) {
            Some(v) => collector / v,
            _ => panic!("Tensor size() shitted its pants"),
        }
    }

    fn max(&self) -> Self::Output {
        let mut m: T = T::min_value();
        self.iter().for_each(|component| {
            if *component.value > m {
                m = *component.value;
            }
        });

        m
    }

    fn min(&self) -> Self::Output {
        let mut m: T = Float::max_value();
        self.iter().for_each(|component| {
            if *component.value < m {
                m = *component.value;
            }
        });

        m
    }
}

/// A single component of a tensor, usually returned by iterators at
/// each iteration
#[derive(Debug)]
pub struct TensorComponent<'a, T> {
    pub coords: Coord,
    pub value: &'a T,
    pub tensor: &'a Tensor<T>,
}

impl<'a, T> TensorComponent<'a, T> {
    pub fn is_terminal(&self) -> bool {
        let higher_dims = self.tensor.shape.range(0, self.tensor.shape.len() - 1);
        self.coords
            .range(0, self.coords.cardinality() - 1)
            .eq(&higher_dims)
    }
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

impl<T: Num + Copy> Dot<Vec<TensorComponent<'_, T>>> for Vec<TensorComponent<'_, T>> {
    type Output = T;

    fn dot(&self, rhs: Vec<TensorComponent<T>>) -> Self::Output {
        let mut acc: Self::Output = T::zero();
        for i in 0..self.len() {
            acc = acc + (*self[i].value * *rhs[i].value);
        }
        acc
    }
}

/// Implements an iteration over a tensor. The iteration is performed
/// according to the underlying coordinate iteration. See `CoordIterator`.
pub struct TensorIterator<'a, T> {
    tensor: &'a Tensor<T>,
    coord_iter: CoordIterator<'a>,
}

impl<'a, T> TensorIterator<'a, T> {
    pub fn new(tensor: &'a Tensor<T>) -> Self {
        Self {
            tensor,
            coord_iter: CoordIterator::new(tensor.shape_ref()),
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

/// Same as `TensorIterator`, but mutable.
pub struct TensorIteratorMut<'a, T> {
    tensor: *mut Tensor<T>,
    coord_iter: CoordIterator<'a>,
}

impl<'a, T> TensorIteratorMut<'a, T> {
    pub fn new(tensor: &'a mut Tensor<T>) -> Self {
        let t: *mut Tensor<T> = &mut *tensor;
        let t_shape = tensor.shape_ref();
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

#[derive(Debug, PartialEq)]
pub enum EnumerationPoint<'a, T> {
    AxisBegin,
    AxisEnd,
    Terminal(TensorComponent<'a, T>),
}

pub struct TensorEnumerator<'a, T> {
    tensor: &'a Tensor<T>,
    current_coord: Option<Coord>,
    start_flags: Vec<bool>,
}

impl<'a, T> TensorEnumerator<'a, T> {
    pub fn new(source: &'a Tensor<T>) -> Self {
        let mut start_flags = Vec::new();
        let size: usize = match source.shape().size() {
            0 => 1,
            _ => source.shape().size(),
        };

        for _ in 0..size {
            start_flags.push(false);
        }

        TensorEnumerator {
            tensor: source,
            current_coord: None,
            start_flags,
        }
    }

    fn step(&mut self) -> Option<EnumerationPoint<'a, T>> {
        if let Some(current_coord) = self.current_coord.as_mut() {
            if current_coord.empty() {
                // If current coordinate is empty it means that the higher order axis has been popped,
                // which means it has been walked and thus the iteration is over
                None
            } else {
                // Otherwise, it means there are still coordinates to be walked
                let space = self.tensor.shape();
                let current_axis_index = current_coord.size() - 1;

                // Check the start flags to know if this axis should increment or not
                if !self.start_flags[current_axis_index] {
                    // SCENARIO: this axis was added on previous iteration, set as started before evaluating it
                    self.start_flags[current_axis_index] = true;
                } else {
                    // SCENARIO: axis index can be advanced (if possible)
                    if current_coord[current_axis_index] < space[current_axis_index] - 1 {
                        // Axis can be advanced
                        current_coord[current_axis_index] += 1;
                    } else {
                        // Axis is done, pop it and reset the start flag to false
                        current_coord.pop_axis();
                        self.start_flags[current_axis_index] = false;
                        return Some(EnumerationPoint::AxisEnd);
                    }
                }

                if current_axis_index < space.size() - 1 {
                    // SCENARIO: this is a non terminal axis, meaning **we must go deeper** and start
                    // a new one
                    current_coord.add_axis();
                    Some(EnumerationPoint::AxisBegin)
                } else {
                    // SCENARIO: this is a terminal axis, meaning there is a value to return
                    Some(EnumerationPoint::Terminal(TensorComponent {
                        coords: current_coord.clone(),
                        tensor: self.tensor,
                        value: &self.tensor[current_coord.clone()],
                    }))
                }
            }
        } else {
            // Iteration has not started yet

            // If the tensor is scalar, handle it here
            if self.tensor.is_scalar() {
                if self.start_flags[0] {
                    None
                } else {
                    self.start_flags[0] = true;
                    Some(EnumerationPoint::Terminal(TensorComponent {
                        coords: coord!(0),
                        tensor: &self.tensor,
                        value: self.tensor.first().unwrap(),
                    }))
                }
            } else {
                // Otherwise, se the start of the coordinate system and step once again
                self.current_coord = Some(Coord::zeroes(1));
                Some(EnumerationPoint::AxisBegin)
            }
        }
    }
}

impl<'a, T> Iterator for TensorEnumerator<'a, T> {
    type Item = EnumerationPoint<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.step()
    }
}

#[macro_export]
/// Constructs a `Tensor`
/// ```ignore
/// tensor!((2, 2, 3) => [3, 3, 1, 2, 2, 2, 1, 1, 4, 5, 2, 4])
/// ```
macro_rules! tensor {
    ( ( $( $dim:expr ),* ) => [ $( $x:expr ),* ] ) => {
        {
            let mut shape_dimensions = Vec::<usize>::new();
            $(
                shape_dimensions.push($dim);
            )*
            let shape = Shape::new(shape_dimensions);

            let mut values = Vec::new();
            $(
                values.push($x);
            )*

            if shape.cardinality() != values.len() {
                panic!(
                    "Trying to create a tensor with shape {} (cardinality {}) but {} values where provided",
                    shape,
                    shape.cardinality(),
                    values.len()
                );
            }

            Tensor::from_vec_with_shape(
                values,
                shape
            )
        }
    };
}

#[cfg(test)]
mod test {
    use crate::{coord, tensor};

    use super::*;
    #[test]
    fn new() {
        let t = Tensor::new(shape!(2, 4, 3), |coord: Coord, _| {
            coord.cardinality() as f64
        });
        assert_eq!(t.shape_ref(), &shape!(2, 4, 3));
    }

    #[test]
    fn zeros() {
        let t: Tensor<f64> = Tensor::zeros(shape!(2, 3, 4, 5));
        assert_eq!(t.rank(), 4);
        assert_eq!(t.size(), 120);
        assert!(t.iter().all(|e| { e.value == &0f64 }));
    }

    #[test]
    fn ones() {
        let t: Tensor<f64> = Tensor::ones(shape!(2, 3, 4, 5));
        assert_eq!(t.rank(), 4);
        assert_eq!(t.size(), 120);
        assert!(t.iter().all(|e| { e.value == &1f64 }));
    }

    #[test]
    fn at() {
        let generator = |coord: Coord, _| coord.cardinality() as f64;
        let t = Tensor::new(shape!(2, 4, 3), generator);
        assert_eq!(t.shape_ref(), &shape!(2, 4, 3));
        let test_value = t.at(coord!(1, 2, 2));
        assert_eq!(test_value, Some(&4.0));
    }

    #[test]
    fn iter_mut() {
        let generator = |coord: Coord, _| coord.cardinality() as f64;
        let mut t = Tensor::<f64>::new(shape!(2, 4, 3), generator);
        let c = t.iter_mut().count();
        assert_eq!(c, 24);
    }

    #[test]
    fn enumerate_scalar() {
        let scalar = Tensor::scalar(1);
        let mut iter = scalar.enumerate();
        assert_eq!(
            iter.next(),
            Some(EnumerationPoint::Terminal(TensorComponent {
                coords: coord!(0),
                tensor: &scalar,
                value: &1
            }))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn enumerate() {
        let tensor_1 = tensor!((2, 3, 2) => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let mut iter = tensor_1.enumerate();
        let mut c = iter.next().unwrap();
        assert_eq!(c, EnumerationPoint::AxisBegin);

        c = iter.next().unwrap();
        assert_eq!(c, EnumerationPoint::AxisBegin);

        c = iter.next().unwrap();
        assert_eq!(c, EnumerationPoint::AxisBegin);

        c = iter.next().unwrap();
        assert_eq!(
            c,
            EnumerationPoint::Terminal(TensorComponent {
                tensor: &tensor_1,
                value: &0,
                coords: coord!(0, 0, 0)
            })
        );

        c = iter.next().unwrap();
        assert_eq!(
            c,
            EnumerationPoint::Terminal(TensorComponent {
                tensor: &tensor_1,
                value: &1,
                coords: coord!(0, 0, 1)
            })
        );

        c = iter.next().unwrap();
        assert_eq!(c, EnumerationPoint::AxisEnd);

        // etc...
    }

    #[test]
    fn scalar_sum() {
        let generator = |coord: Coord, _| coord.cardinality() as f64;
        let mut t = Tensor::<f64>::new(shape!(2, 4, 3), generator);
        t = t + 2.0;
        assert_eq!(t.at(coord!(1, 2, 2)), Some(&6.0));
    }

    #[test]
    fn scalar_mul() {
        let generator = |coord: Coord, _| coord.cardinality() as f64;
        let mut t = Tensor::<f64>::new(shape!(2, 4, 3), generator);
        t = t * 2.0;
        assert_eq!(t.at(coord!(1, 2, 2)), Some(&8.0));
    }

    #[test]
    fn ordering() {
        let generator = |_, i| i as u64;
        let t: Tensor<u64> = Tensor::new(shape!(2, 4, 3), generator);

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
        assert_ne!(t.set(coord!(0, 0, 0), 8), Err(TensorError::Set));
        assert_eq!(t[coord!(0, 0, 0)], 8);
        assert_eq!(t.set(coord!(4, 1, 10), 10), Err(TensorError::NoCoordinate));
    }

    #[test]
    fn uninit_tensor_no_panic() {
        let t = Tensor::<u8>::new_uninit(shape!(4, 5, 6));
        assert!(t.at(coord!(0, 1, 3)).is_some());
    }

    #[test]
    fn stats() {
        let generator = |_coord, i| i as f64;
        let t = Tensor::<f64>::new(shape!(2, 4, 3), generator);
        assert_eq!(t.size(), 24);
        assert_eq!(t.mean(), 11.5);
        assert_eq!(t.max(), 23.0);
        assert_eq!(t.min(), 0.0);
    }

    #[test]
    fn display() {
        let t: Tensor<f64> = Tensor::zeros(shape!(4, 3, 2));
        println!("{}", t);
    }

    #[test]
    fn from_vec() {
        let tensor1 = Tensor::from_vec_with_shape(vec![1, 2, 3, 4], shape!(2, 2));
        assert_eq!(tensor1.size(), 4);
    }

    #[test]
    fn rescale() {
        let mut tensor1 = tensor!((1, 3, 2) => [ 0., 1., 4., 3.2, 3.1, 9. ]);
        assert_eq!(tensor1.shape(), shape!(1, 3, 2));

        tensor1.reshape(shape!(2, 3, 1));
        assert_eq!(tensor1.shape(), shape!(2, 3, 1));

        tensor1.reshape(shape!(6, 1, 1));
        assert_eq!(tensor1.shape(), shape!(6, 1, 1));

        tensor1.reshape(shape!(6));
        assert_eq!(tensor1.shape(), shape!(6));
    }

    #[test]
    fn macros() {
        let tensor1 = tensor!((1, 3, 2) => [ 0., 1., 4., 3.2, 3.1, 9. ]);
        assert_eq!(tensor1.size(), 6);
        assert_eq!(tensor1.at(coord!(0, 0, 0)), Some(&0.));
        assert_eq!(tensor1.at(coord!(0, 0, 1)), Some(&1.));
        assert_eq!(tensor1.at(coord!(0, 1, 0)), Some(&4.));
        assert_eq!(tensor1.at(coord!(0, 1, 1)), Some(&3.2));
        assert_eq!(tensor1.at(coord!(0, 2, 0)), Some(&3.1));
        assert_eq!(tensor1.at(coord!(0, 2, 1)), Some(&9.));
    }

    #[test]
    #[should_panic]
    fn macros_panic() {
        let _tensor1 = tensor!((1, 3, 2) => [ 0., 1., 4. ]);
    }
}
