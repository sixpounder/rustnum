use std::{
    fmt::Display,
    ops::{Index, IndexMut},
    slice::Iter,
    slice::IterMut
};

use super::{Set, Shape};

/// Represents a generic coordinate for a given space
#[derive(Clone, Debug)]
pub struct Coord {
    axis: Vec<usize>,
}

impl Coord {
    /// Creates a new coordinate with `axis` as its components
    #[inline]
    pub fn new(axis: Vec<usize>) -> Self {
        Self { axis: axis }
    }

    #[inline]
    pub fn new_for_shape(space: Shape) -> Self {
        Self::zeroes(space.size())
    }

    /// Creates a new coordinate of `n_dimensions` component set to `0`
    #[inline]
    pub fn zeroes(n_dimensions: usize) -> Self {
        let mut dimensions = Vec::with_capacity(n_dimensions);
        for _ in 0..n_dimensions {
            dimensions.push(0);
        }

        Self { axis: dimensions }
    }

    #[inline]
    pub fn terminal(space: Shape) -> Self {
        let mut coord = Self::new_for_shape(space.clone());
        let mut i = 0;
        for axis in coord.iter_axis_mut() {
            *axis = space[i] - 1;
            i += 1;
        }

        coord
    }

    #[inline]
    pub fn range(&self, start: usize, end: usize) -> Self {
        let dimensions = self.axis[start..end].to_vec();
        Self { axis: dimensions }
    }

    /// Gets the value of the axis component at `idx` position.
    /// The same can be done with regular indexing (like `my_coord[0]`)
    #[inline]
    pub fn get_axis(&self, idx: usize) -> Option<&usize> {
        self.axis.get(idx)
    }

    /// `get_axis` mutable output variant
    #[inline]
    pub fn get_axis_mut(&mut self, idx: usize) -> Option<&mut usize> {
        self.axis.get_mut(idx)
    }

    #[inline]
    pub fn add_axis(&mut self) -> usize {
        self.axis.push(0);
        self.axis.len() - 1
    }

    #[inline]
    pub fn remove_axis(&mut self, idx: usize) {
        self.axis.remove(idx);
    }

    #[inline]
    pub fn pop_axis(&mut self) -> Option<usize> {
        self.axis.pop()
    }

    /// An iterator over the coordinate components
    #[inline]
    pub fn iter_axis(&self) -> Iter<'_, usize> {
        self.axis.iter()
    }

    /// An iterator over the coordinate components
    #[inline]
    pub fn iter_axis_mut(&mut self) -> IterMut<'_, usize> {
        self.axis.iter_mut()
    }

    /// The cardinality of the coordinate (the multiplication of all its components)
    #[inline]
    pub fn cardinality(&self) -> usize {
        let mut cardinality = 1;
        self.axis.iter().for_each(|i| {
            cardinality = cardinality * i;
        });
        cardinality
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.size() == 0
    }
}

impl Set for Coord {
    type Item = usize;

    fn at(&self, idx: usize) -> Option<&usize> {
        self.get_axis(idx)
    }

    fn size(&self) -> usize {
        self.axis.len()
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

impl<S> PartialEq<S> for Coord
where
    S: Set<Item = usize>,
{
    fn eq(&self, other: &S) -> bool {
        self.as_set_slice() == other.as_set_slice()
    }
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string_list: Vec<String> = self.iter_axis().map(|axis| axis.to_string()).collect();
        write!(f, "({})", string_list.join(","))
    }
}

/// An iterator over a coordinate system in a space described by a `shape`
#[derive(Debug)]
pub struct CoordIterator<'a> {
    space: &'a Shape,
    current: Coord,
    previous: Option<Coord>,
    next: Option<Coord>,
    started: bool,
}

impl<'a> CoordIterator<'a> {
    /// Creates a new coordinate iterator for a given `Shape`, starting at the shape origin
    /// # Example
    /// ```
    /// # use rustnum::prelude::*;
    /// # use rustnum::{shape};
    /// let s = shape!(3, 5, 3);
    /// let mut iter_one = s.iter().take(1);
    /// assert_eq!(iter_one.next(), Some(Coord::new(vec![0, 0, 0])));
    /// ```
    pub fn new(space: &'a Shape) -> Self {
        Self {
            space,
            current: Coord::zeroes(space.len()),
            previous: None,
            next: None,
            started: false,
        }
    }

    pub fn from_mut(space: &'a mut Shape) -> Self {
        Self {
            space,
            current: Coord::zeroes(space.len()),
            previous: None,
            next: None,
            started: false,
        }
    }

    pub fn has_next(&self) -> bool {
        self.next.is_some()
    }

    pub fn step(&mut self) {
        if !self.started {
            self.started = true;
            self.previous = None;
            self.next = Some(self.current.clone());
        } else {
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
}

impl<'a> Iterator for CoordIterator<'a> {
    type Item = Coord;

    fn next(&mut self) -> Option<Self::Item> {
        self.step();
        if self.has_next() {
            Some(self.next.clone().unwrap())
        } else {
            None
        }
    }
}

#[macro_export]
macro_rules! coord {
    () => {
        Coord::new(vec![])
    };
    ( $( $x:expr ),* ) => {
        {
            let mut dims = Vec::<usize>::new();
            $(
                dims.push($x);
            )*
            let o = Coord::new(dims);
            o
        }
    };
}

#[cfg(test)]
mod test {
    #[test]
    fn base() {
        use super::*;
        use crate::shape;

        let s = shape!(3, 5, 3);
        let mut iter_one = s.iter().take(1);
        assert_eq!(iter_one.next(), Some(Coord::new(vec![0, 0, 0])));
    }
}
