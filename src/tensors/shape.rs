use std::ops::{Index, IndexMut};
use std::slice::Iter;

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

    pub fn next_coord(shape: &Shape, from: &Shape) -> Option<Shape> {
        let mut end_of_shape = false;
        let mut coord = from.clone();
        let shape_last_index = shape.len() - 1;
        let mut i = shape_last_index;
        while let Some(coordinate) = coord.get_axis_mut(i) {
            if *coordinate < (shape[i] - 1) {
                // Can move on this axis
                *coordinate = *coordinate + 1;
    
                // If the axis is NOT the last one, we must reset all contained axis
                if i < shape_last_index {
                    let reset_rng = (i + 1)..=shape_last_index;
                    for j in reset_rng {
                        coord[j] = 0;
                    }
                }
    
                break;
            } else {
                if i == 0 {
                    // Axis is done and there are no more axis, shape ended
                    end_of_shape = true;
                    break;
                } else {
                    // Axis is done, check the next one
                    i -= 1;
                    continue;
                }
            }
        }
    
        if end_of_shape {
            None
        } else {
            Some(from.clone())
        }
    }

    pub fn empty() -> Self {
        Self {
            dimensions: vec![],
            scale_factors: vec![],
        }
    }

    pub fn zeroes(n_dimensions: usize) -> Self {
        let mut dimensions = Vec::with_capacity(n_dimensions);
        for i in 0..n_dimensions {
            dimensions.push(0);
        }

        let scale_factors = compute_scale_factors(&dimensions);

        Self {
            dimensions,
            scale_factors,
        }
    }

    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    pub fn first(&self) -> Option<&usize> {
        self.dimensions.first()
    }

    pub fn last(&self) -> Option<&usize> {
        self.dimensions.last()
    }

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

    pub fn iter_axis(&self) -> Iter<'_, usize> {
        self.dimensions.iter()
    }

    pub fn get_axis(&self, idx: usize) -> Option<&usize> {
        self.dimensions.get(idx)
    }

    pub fn get_axis_mut(&mut self, idx: usize) -> Option<&mut usize> {
        self.dimensions.get_mut(idx)
    }

    pub fn mul(&self) -> usize {
        let mut p = 1;
        self.dimensions.iter().for_each(|i| {
            p = p * i;
        });
        p
    }

    pub fn axis_cardinality(&self, axis: usize) -> Option<&usize> {
        self.scale_factors.get(axis)
    }

    pub fn equiv(&self, other: &Self) -> bool {
        self.mul() == other.mul()
    }

    pub fn iter(&self) -> ShapeIterator {
        ShapeIterator {
            shape: self,
            started: true,
            curr_coord: Shape::zeroes(self.len()),
            next_coord: Some(Shape::zeroes(self.len()))
        }
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

pub struct ShapeIterator<'a> {
    shape: &'a Shape,
    started: bool,
    curr_coord: Shape,
    next_coord: Option<Shape>
}

impl<'a> ShapeIterator<'a> {
    fn has_next(&self) -> bool {
        self.next_coord.is_some()
    }

    fn step_forward(&mut self) {
        if self.started {
            self.started = false;
        } else {
            let mut end_of_shape = false;
            let shape_last_index = self.shape.len() - 1;
            let mut i = shape_last_index;
            while let Some(coordinate) = self.curr_coord.get_axis_mut(i) {
                if *coordinate < (self.shape[i] - 1) {
                    // Can move on this axis
                    *coordinate = *coordinate + 1;
    
                    // If the axis is NOT the last one, we must reset all contained axis
                    if i < shape_last_index {
                        let reset_rng = (i + 1)..=shape_last_index;
                        for j in reset_rng {
                            self.curr_coord[j] = 0;
                        }
                    }
    
                    break;
                } else {
                    if i == 0 {
                        // Axis is done and there are no more axis, shape ended
                        end_of_shape = true;
                        break;
                    } else {
                        // Axis is done, check the next one
                        i -= 1;
                        continue;
                    }
                }
            }
    
            if end_of_shape {
                self.next_coord = None;
            } else {
                self.next_coord = Some(self.curr_coord.clone());
            }
        }
    }
}

impl<'a> Iterator for ShapeIterator<'a> {
    type Item = Shape;

    fn next(&mut self) -> Option<Self::Item> {
        self.step_forward();
        if !self.has_next() {
            None
        } else {
            let next = self.next_coord.as_ref().unwrap();
            Some(
                next.clone()
            )
        }
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
    use crate::{shape};

    use super::*;

    #[test]
    pub fn scale_factors() {
        let s = shape!(3, 4);
        assert_eq!(s.scale_factors[0], 4);
        assert_eq!(s.scale_factors[1], 1);
    }

    #[test]
    pub fn coordinates_iter() {
        let mut s = shape!(3, 4, 9);
        let mut all_coords: Vec<Shape> = s.iter().collect();
        assert_eq!(all_coords.len(), 108);
    }
}
