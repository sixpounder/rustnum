use std::ops::Index;

#[derive(Clone)]
pub struct SetSlice<'a, T> {
    head: *const T,
    n: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T> SetSlice<'a, T> {
    pub fn new(head: &'a T, start: usize, end: usize) -> Self {
        Self {
            head: head as *const T,
            n: end - start,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn at(&self, index: usize) -> Option<&'a T> {
        if index < self.n {
            unsafe { Some(&*self.head.add(index)) }
        } else {
            None
        }
    }

    pub fn at_unchecked(&self, index: usize) -> &'a T {
        self.at(index).unwrap()
    }
}

impl<T> SetSlice<'_, T> {
    pub fn size(&self) -> usize {
        self.n
    }
}

impl<T> Index<usize> for SetSlice<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.at_unchecked(index)
    }
}

impl<T: PartialEq> PartialEq for SetSlice<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { *self.head == *other.head && self.n == other.n }
    }
}

impl<'a, T> IntoIterator for SetSlice<'a, T> {
    type Item = &'a T;

    type IntoIter = SetIntoIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SetIntoIterator::from_slice(self)
    }
}

pub struct SetIntoIterator<'a, T> {
    slice: SetSlice<'a, T>,
    i: usize,
}

impl<'a, T> SetIntoIterator<'a, T> {
    pub fn from_slice(slice: SetSlice<'a, T>) -> Self {
        Self { slice, i: 0 }
    }
}

impl<'a, T> Iterator for SetIntoIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.slice.size() {
            let returned = self.slice.at(self.i);
            self.i += 1;
            returned
        } else {
            None
        }
    }
}

pub trait Set {
    type Item;

    /// Returns the item with index `idx` in this set or `None` if not found
    fn at(&self, idx: usize) -> Option<&Self::Item>;

    /// The number of items in this set
    fn size(&self) -> usize;

    /// Same as `at` but without panic check
    fn at_unchecked(&self, idx: usize) -> &Self::Item {
        self.at(idx).unwrap()
    }

    /// Gets a slice of this set from `start` to `end` (exclusive)
    fn slice(&self, start: usize, end: usize) -> SetSlice<'_, Self::Item> {
        SetSlice::new(self.at_unchecked(0), start, end)
    }

    fn as_set_slice(&self) -> SetSlice<'_, Self::Item> {
        SetSlice::new(self.at_unchecked(0), 0, self.size())
    }

    fn enumerate(&self) -> SetIntoIterator<Self::Item> {
        self.as_set_slice().into_iter()
    }
}

impl<T: PartialEq> PartialEq for dyn Set<Item = T> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..self.size() {
            if other.at_unchecked(i) != self.at_unchecked(i) {
                return false;
            }
        }

        true
    }
}

impl<T> Set for Vec<T> {
    type Item = T;

    fn at(&self, idx: usize) -> Option<&T> {
        self.get(idx)
    }

    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod test {
    use super::Set;

    #[test]
    fn slice() {
        let set1 = vec![1, 2, 3, 4];
        let slice = set1.slice(0, 2);
        assert_eq!(slice.size(), 2);
    }

    #[test]
    fn iterator() {
        let set1 = vec![1, 2, 3, 4];
        let mut iter = set1.enumerate();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn sets_equality() {
        let set1: Vec<u8> = vec![1, 2, 3, 4];
        let set2: Vec<u8> = vec![1, 2, 3, 4];

        assert!(set1.as_set_slice() == set2.as_set_slice());

        let set1: Vec<u8> = vec![1, 2, 3, 4];
        let set2: Vec<u8> = vec![1, 4, 8, 16];

        assert_eq!(set1.as_set_slice() != set2.as_set_slice(), false);
    }
}
