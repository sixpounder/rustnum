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

    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    pub fn last(&self) -> Option<&usize> {
        self.dimensions.last()
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

    pub fn iter(&self) -> Iter<'_, usize> {
        self.dimensions.iter()
    }

    pub fn mul(&self) -> usize {
        let mut p = 1;
        self.dimensions.iter().for_each(|i| { p *= i });
        p
    }

    pub fn equiv(&self, other: &Self) -> bool {
        self.mul() == other.mul()
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, idx: usize) -> &<Self as std::ops::Index<usize>>::Output {
        &self.dimensions[idx]
    }
}
