use std::slice::Iter;
use std::ops::Index;

#[derive(Debug)]
pub struct Shape {
    dimensions: Vec<usize>
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Self {
            dimensions: vec
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(src: &[usize]) -> Self {
        Self {
            dimensions: Vec::from(src)
        }
    }
}

impl PartialEq<Self> for Shape {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..self.dimensions.len() {
            if self.dimensions[i] != other.dimensions[i] {
                return false;
            } else {
                continue
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
                continue
            }
        }

        true
    }
}

impl Clone for Shape {
    fn clone(&self) -> Self {
        let mut dimensions: Vec<usize> = vec![];
        self.dimensions.iter().for_each(|dim| { dimensions.push(*dim); });

        Self {
            dimensions
        }
    }
}

impl Eq for Shape {}

impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self::from(dimensions)
    }

    pub fn empty() -> Self {
        Self {
            dimensions: vec![]
        }
    }

    pub fn zeroes(n_dimensions: usize) -> Self {
        let mut dimensions = Vec::with_capacity(n_dimensions);
        for i in 0..n_dimensions {
            dimensions.push(0);
        }

        Self {
            dimensions
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

    pub fn tail(&self) -> Shape {
        let slice: &[usize];
        if self.len() > 1 {
            slice = self.dimensions[1..self.dimensions.len()].as_ref();
        } else {
            slice = &[];
        }
        Shape::from(slice)
    }

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
        self.dimensions.iter().for_each(|i| { p = p * i; });
        p
    }

    pub fn pos(&self) -> usize {
        0
    }

    pub fn equiv(&self, other: &Self) -> bool {
        self.mul() == other.mul()
    }

    pub fn iter(&self) -> ShapeIterator {
        ShapeIterator {
            shape: self,
            curr: Shape::zeroes(self.len()),
            next: Shape::zeroes(self.len()),
        }
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, idx: usize) -> &<Self as std::ops::Index<usize>>::Output {
        &self.dimensions[idx]
    }
}

pub struct ShapeIterator<'a> {
    shape: &'a Shape,
    curr: Shape,
    next: Shape
}

pub struct ShapeIter<'a, T> {
    pub coords: &'a Shape,
    pub value: T
}

impl<'a> Iterator for ShapeIterator<'a> {
    type Item = ShapeIter<'a, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(ShapeIter {
            value: 0,
            coords: self.shape
        })
    }
}
