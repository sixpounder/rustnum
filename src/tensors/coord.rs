use std::ops::{Index, IndexMut};

use crate::Shape;

#[derive(Clone)]
pub struct Coord {
    axis: Vec<usize>,
}

impl Coord {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self {
            axis: dimensions,
        }
    }

    pub fn zeroes(n_dimensions: usize) -> Self {
        let mut dimensions = Vec::with_capacity(n_dimensions);
        for _ in 0..n_dimensions {
            dimensions.push(0);
        }

        Self {
            axis: dimensions
        }
    }

    pub fn get_axis(&self, idx: usize) -> Option<&usize> {
        self.axis.get(idx)
    }

    pub fn get_axis_mut(&mut self, idx: usize) -> Option<&mut usize> {
        self.axis.get_mut(idx)
    }
}

impl Index<usize> for Coord {
    type Output = usize;
    fn index(&self, idx: usize) -> &<Self as std::ops::Index<usize>>::Output {
        &self.axis[idx]
    }
}

impl IndexMut<usize> for Coord {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.axis[idx]
    }
}

pub struct CoordSystem<'a> {
    space: &'a Shape,
    current: Coord,
    next: Option<Coord>
}

impl<'a> CoordSystem<'a> {
    pub fn new(space: &'a Shape) -> Self {
        Self {
            space,
            current: Coord::zeroes(space.len()),
            next: None
        }
    }

    pub fn step(&mut self) {
        let mut end_of_shape = false;
        let shape_last_index = self.space.len() - 1;
        let mut i = shape_last_index;
        while let Some(coordinate) = self.current.get_axis_mut(i) {
            if *coordinate < (self.space[i] - 1) {
                // Can move on this axis
                *coordinate = *coordinate + 1;

                // If the axis is NOT the last one, we must reset all contained axis
                if i < shape_last_index {
                    let reset_rng = (i + 1)..=shape_last_index;
                    for j in reset_rng {
                        self.current[j] = 0;
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
            self.next = None;
        } else {
            self.next = Some(self.current.clone());
        }
    }
}

impl<'a> Iterator for CoordSystem<'a> {
    type Item = Coord;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
