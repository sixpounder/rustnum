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

impl Eq for Shape {}

impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self::from(dimensions)
    }

    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    pub fn last(&self) -> Option<&usize> {
        self.dimensions.last()
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
