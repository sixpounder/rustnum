use std::ops::{Index, IndexMut};
use std::slice::Iter;
use crate::{CoordIterator};

/// A shape is a description of a space with `n` independent dimensions
#[derive(Debug)]
pub struct Shape {
    dimensions: Vec<usize>,
    scale_factors: Vec<usize>,
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

impl PartialEq<Self> for Shape {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..self.dimensions.len() {
            if self.dimensions[i] != other.dimensions[i] {
                return false;
            } else {
                continue;
            }
        }

        true
    }
}

impl PartialEq<Vec<usize>> for Shape {
    fn eq(&self, other: &Vec<usize>) -> bool {
        for i in 0..self.dimensions.len() {
            if self.dimensions[i] != other[i] {
                return false;
            } else {
                continue;
            }
        }

        true
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

impl Eq for Shape {}

impl Shape {
    /// Creates a new shape with some `dimensions`
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

    /// Creates an empty shape
    pub fn empty() -> Self {
        Self {
            dimensions: vec![],
            scale_factors: vec![],
        }
    }

    /// Creates a shape with `n_dimensions` set to 0
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
    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the first dimension of this shape
    pub fn first(&self) -> Option<&usize> {
        self.dimensions.first()
    }

    /// Returns the last dimension of this shape
    pub fn last(&self) -> Option<&usize> {
        self.dimensions.last()
    }

    /// Returns `true` if some `other` shape is contained by this shape
    pub fn includes(&self, other: &Shape) -> bool {
        for i in 0..self.dimensions.len() {
            if self.dimensions[i] < other.dimensions[i] {
                return false;
            }
        }

        true
    }

    /// Returns a new `Shape` with all but the first axis
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
    pub fn head(&self) -> Shape {
        if self.len() > 0 {
            Shape::new(vec![self.dimensions[0]])
        } else {
            Shape::empty()
        }
    }

    pub fn replace_last(&self, value: usize) -> Shape {
        let mut t = self.tail();
        t.dimensions.push(value);

        t
    }

    /// Appends an axis to this shape
    pub fn append(&mut self, value: usize) {
        self.dimensions.push(value);
        self.scale_factors = compute_scale_factors(&self.dimensions);
    }

    /// Returns an iterator over the shape's axis
    pub fn iter_axis(&self) -> Iter<'_, usize> {
        self.dimensions.iter()
    }

    /// Get the axis with index `idx`
    pub fn get_axis(&self, idx: usize) -> Option<&usize> {
        self.dimensions.get(idx)
    }

    /// Get the axis with index `idx` as mutable
    pub fn get_axis_mut(&mut self, idx: usize) -> Option<&mut usize> {
        self.dimensions.get_mut(idx)
    }

    /// The cardinality of this shape
    pub fn mul(&self) -> usize {
        let mut p = 1;
        self.dimensions.iter().for_each(|i| {
            p = p * i;
        });
        p
    }

    /// The cardinality of a single axis in this shape
    pub fn axis_cardinality(&self, axis: usize) -> Option<&usize> {
        self.scale_factors.get(axis)
    }

    /// Returns 'true` if some `other` shape has the same cardinality as this shape
    pub fn equiv(&self, other: &Self) -> bool {
        self.mul() == other.mul()
    }

    /// An iterator over all the coordinates contained by this shape
    pub fn iter(&self) -> CoordIterator {
        CoordIterator::new(self)
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, idx: usize) -> &<Self as std::ops::Index<usize>>::Output {
        &self.dimensions[idx]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.dimensions[idx]
    }
}

/// Utility to compute axis cardinalities for later use
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
}
