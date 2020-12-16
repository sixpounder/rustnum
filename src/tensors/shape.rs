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
        let scale_factors = compute_scale_factors(&dimensions);
        Self {
            dimensions,
            scale_factors,
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

    pub fn append(&mut self, value: usize) {
        self.dimensions.push(value);
    }

    pub fn iter_axis(&self) -> Iter<'_, usize> {
        self.dimensions.iter()
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
            curr_coord: Shape::zeroes(self.len()),
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
    curr_coord: Shape,
}

pub struct ShapeIter<'a> {
    pub coords: &'a Shape
}

impl<'a> Iterator for ShapeIterator<'a> {
    type Item = ShapeIter<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut out_value = None;
        for i_rev in (self.shape.dimensions.len() - 1)..0 {
            let current_axis_value = self.curr_coord[i_rev];
            if current_axis_value < self.shape[i_rev] {
                self.curr_coord[i_rev] = self.curr_coord[i_rev] + 1;
                break;
            }
        }
        out_value = Some(ShapeIter {
            coords: self.shape,
        });

        out_value
    }
}

fn compute_scale_factors(dimensions: &Vec<usize>) -> Vec<usize> {
    let mut scale_factors = Vec::with_capacity(dimensions.len());
    let mut i = 0;
    while i < dimensions.len() {
        if i == dimensions.len() - 1 {
            // Last item
            scale_factors.push(1);
        } else {
            let rng = i..dimensions.len();
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
