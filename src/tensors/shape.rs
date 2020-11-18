use std::ops::Index;

#[derive(Debug)]
pub struct Shape {
    dimensions: Vec<usize>
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self { todo!() }
}

impl PartialEq for Shape {
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

impl Eq for Shape {}

impl Shape {
    pub fn len(&self) -> usize {
        self.dimensions.len()
    }

    pub fn last(&self) -> Option<&usize> {
        self.dimensions.last()
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, idx: usize) -> &<Self as std::ops::Index<usize>>::Output {
        &self.dimensions[idx]
    }
}