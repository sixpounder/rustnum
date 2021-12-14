use rand::Rng;

use crate::{CoordIterator, Set, Coord};
use std::slice::Iter;
use std::{
    fmt::Display,
    ops::{Index, IndexMut, Mul},
};

pub trait Shapeable {
    fn shape() -> Shape;
}

/// A shape is a description of a space with `n` independent dimensions
#[derive(Debug)]
pub struct Shape {
    dimensions: Vec<usize>,
    scale_factors: Vec<usize>,
}

impl Set for Shape {
    type Item = usize;
    fn at(&self, idx: usize) -> Option<&usize> {
        self.get_axis(idx)
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl Shape {
    /// Creates a new shape with some `dimensions`
    #[inline]
    pub fn new(dimensions: Vec<usize>) -> Self {
        if dimensions.iter().any(|f| *f == 0) {
            panic!("Zero value dimensions are not allowed in shapes");
        }

        let scale_factors = compute_scale_factors(&dimensions);
        Self {
            dimensions,
            scale_factors,
        }
    }

    #[inline]
    pub fn scalar() -> Self {
        let dimensions = vec![];
        let scale_factors = compute_scale_factors(&dimensions);
        Self {
            dimensions,
            scale_factors,
        }
    }

    /// Creates an empty shape
    #[inline]
    pub fn empty() -> Self {
        Self {
            dimensions: vec![],
            scale_factors: vec![],
        }
    }

    /// Creates a shape with `n_dimensions` set to 0
    #[inline]
    pub fn zeroes(n_dimensions: usize) -> Self {
        let mut dimensions = Vec::with_capacity(n_dimensions);
        for _ in 0..n_dimensions {
            dimensions.push(0);
        }

        let scale_factors = compute_scale_factors(&dimensions);

        Self {
            dimensions,
            scale_factors,
        }
    }

    /// The number of dimension of this shape
    #[inline]
    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.len() == 0
    }

    pub fn axis_scale_factor(&self, axis: usize) -> Option<&usize> {
        self.scale_factors.get(axis)
    }

    /// The number of dimension of this shape
    #[inline]
    pub fn ndim(&self) -> usize {
        self.len()
    }

    #[inline]
    pub fn range(&self, start: usize, end: usize) -> Self {
        let dimensions = self.dimensions[start..end].to_vec();
        let scale_factors = compute_scale_factors(&dimensions);
        Shape {
            dimensions,
            scale_factors,
        }
    }

    /// Gets a random coord inside this shape
    #[inline]
    pub fn random_coord(&self) -> Coord {
        let mut coord = Coord::zeroes(self.ndim());
        let mut rng = rand::thread_rng();
        for dim in 0..self.ndim() {
            coord[dim] = rng.gen_range(0,self.dimensions[dim]);
        }

        coord
    }

    /// Returns the first dimension of this shape
    #[inline]
    pub fn first(&self) -> Option<&usize> {
        self.dimensions.first()
    }

    /// Returns the last dimension of this shape
    #[inline]
    pub fn last(&self) -> Option<&usize> {
        self.dimensions.last()
    }

    /// Returns `true` if some `other` shape is contained by this shape
    #[inline]
    pub fn includes(&self, other: &Shape) -> bool {
        for i in 0..self.dimensions.len() {
            if self.dimensions[i] < other.dimensions[i] {
                return false;
            }
        }

        true
    }

    /// Returns a new `Shape` with all but the first axis
    #[inline]
    pub fn tail(&self) -> Shape {
        let slice: &[usize];
        if self.len() > 1 {
            slice = self.dimensions[1..self.dimensions.len()].as_ref();
        } else {
            slice = &[];
        }
        Shape::from(slice)
    }

    /// Returns a new `Shape` with only the first axis (or an empty one if the original shape is empty)
    #[inline]
    pub fn head(&self) -> Shape {
        if self.len() > 0 {
            Shape::new(vec![self.dimensions[0]])
        } else {
            Shape::empty()
        }
    }

    /// Prepends an axis to this shape
    #[inline]
    pub fn prepend(&mut self, value: usize) {
        self.dimensions.insert(value, 0);
        self.scale_factors = compute_scale_factors(&self.dimensions);
    }

    /// Appends an axis to this shape
    #[inline]
    pub fn append(&mut self, value: usize) {
        self.dimensions.push(value);
        self.scale_factors = compute_scale_factors(&self.dimensions);
    }

    /// Returns an iterator over the shape's axis
    #[inline]
    pub fn iter_axis(&self) -> Iter<'_, usize> {
        self.dimensions.iter()
    }

    /// Get the axis with index `idx`
    #[inline]
    pub fn get_axis(&self, idx: usize) -> Option<&usize> {
        self.dimensions.get(idx)
    }

    /// Get the axis with index `idx` as mutable
    #[inline]
    pub fn get_axis_mut(&mut self, idx: usize) -> Option<&mut usize> {
        self.dimensions.get_mut(idx)
    }

    /// The cardinality of this shape (the multiplication of all its components)
    #[inline]
    pub fn cardinality(&self) -> usize {
        let mut p = 1;
        self.dimensions.iter().for_each(|i| {
            p = p * i;
        });
        p
    }

    /// The cardinality of a single axis in this shape
    #[inline]
    pub fn axis_cardinality(&self, axis: usize) -> Option<&usize> {
        self.scale_factors.get(axis)
    }

    /// Returns `true` if some `other` shape has the same cardinality as this shape. This is usually meant to
    /// be checked to determine if a shape is reshapable into another.
    #[inline]
    pub fn equiv(&self, other: &Self) -> bool {
        self.cardinality() == other.cardinality()
    }

    /// An iterator over all the coordinates contained by this shape
    #[inline]
    pub fn iter(&self) -> CoordIterator {
        CoordIterator::new(self)
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string_list: Vec<String> = self.iter_axis().map(|axis| axis.to_string()).collect();
        write!(f, "({})", string_list.join(","))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Self::new(vec)
    }
}

impl From<&[usize]> for Shape {
    fn from(src: &[usize]) -> Self {
        Self::new(Vec::from(src))
    }
}

impl<S> PartialEq<S> for Shape
where
    S: Set<Item = usize>,
{
    fn eq(&self, other: &S) -> bool {
        self.as_set_slice() == other.as_set_slice()
    }
}

impl Clone for Shape {
    fn clone(&self) -> Self {
        let mut dimensions: Vec<usize> = vec![];
        self.dimensions.iter().for_each(|dim| {
            dimensions.push(*dim);
        });
        let scale_factors = compute_scale_factors(&dimensions);

        Self {
            dimensions,
            scale_factors,
        }
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, idx: usize) -> &<Self as std::ops::Index<usize>>::Output {
        &self.dimensions[idx]
    }
}

impl Mul<Shape> for Shape {
    type Output = Shape;

    fn mul(self, rhs: Shape) -> Self::Output {
        let mut dims: Vec<usize> = vec![];
        for i in 0..self.size() {
            dims[i] = self.dimensions[i] * rhs.dimensions[i];
        }

        let scale_factors = compute_scale_factors(&dims);

        Shape {
            dimensions: dims,
            scale_factors,
        }
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.dimensions[idx]
    }
}

/// Utility to compute axis cardinalities for later use
#[inline]
fn compute_scale_factors(dimensions: &Vec<usize>) -> Vec<usize> {
    let mut scale_factors = Vec::with_capacity(dimensions.len());
    let mut i = 0;
    while i < dimensions.len() {
        if i == dimensions.len() - 1 {
            // Last axis is always 1
            scale_factors.push(1);
        } else {
            let rng = (i + 1)..dimensions.len();
            let mut factor = 1;
            dimensions[rng].iter().for_each(|f| {
                factor *= f;
            });

            scale_factors.push(factor);
        }
        i += 1;
    }

    scale_factors
}

#[macro_export]
macro_rules! shape {
    () => {
        Shape::empty()
    };
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

#[cfg(test)]
mod test {
    use crate::{shape, Coord};

    use super::*;

    #[test]
    pub fn scale_factors() {
        let s = shape!(3, 4);
        assert_eq!(s.scale_factors[0], 4);
        assert_eq!(s.scale_factors[1], 1);
    }

    #[test]
    pub fn coordinates_iter() {
        let s = shape!(3, 4, 9);
        let all_coords: Vec<Coord> = s.iter().collect();
        assert_eq!(all_coords.len(), 108);
    }

    #[test]
    fn shape_equality() {
        let shape_1 = shape!(3, 4, 9);
        let shape_2 = shape!(3, 4, 9);

        assert_eq!(shape_1, shape_2);

        let shape_1 = shape!(3, 4, 9);
        let shape_2 = shape!(1, 2, 3);

        assert!(shape_1 != shape_2)
    }

    #[test]
    fn rand_coord() {
        let shape1 = shape!(3, 4, 5);
        for _ in 0..100 {
            let rand_coord = shape1.random_coord();
            assert!(rand_coord[0] < 3);
            assert!(rand_coord[1] < 4);
            assert!(rand_coord[2] < 5);
        }
    }
}
