use std::ops::Index;

pub struct SetSlice<'a, T: Sized> {
    head: &'a T,
    n: usize,
}

impl<'a, T> SetSlice<'a, T> {
    pub fn new(head: &'a T, start: usize, end: usize) -> Self {
        Self {
            head,
            n: end - start,
        }
    }

    pub fn at(&self, index: usize) -> Option<&'a T> {
        if index < self.n {
            let mut head_addr: *const T = self.head;
            unsafe {
                head_addr = head_addr.add(index);
                Some(&*head_addr)
            }
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

pub struct SetIterator<'a, T> {
    slice: SetSlice<'a, T>,
    i: usize,
}

impl<'a, T> SetIterator<'a, T> {
    pub fn from_slice(slice: SetSlice<'a, T>) -> Self {
        Self {
            slice,
            i: 0,
        }
    }
}

impl<'a, T> Iterator for SetIterator<'a, T> {
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

    fn empty() -> Self;

    /// Returns the item with index `idx` in this set or `None` if not found
    fn at(&self, idx: usize) -> Option<&Self::Item>;

    /// The number of items in this set
    fn size(&self) -> usize;

    /// Gets a slice of this set from `start` to `end` (exclusive)
    fn slice(&self, start: usize, end: usize) -> SetSlice<'_, Self::Item>
    where
        Self: Sized,
    {
        SetSlice::new(self.at_unchecked(0), start, end)
    }

    /// Same as `at` but without panic check
    fn at_unchecked(&self, idx: usize) -> &Self::Item {
        self.at(idx).unwrap()
    }

    fn same(&self, other: &Self) -> bool where Self::Item: PartialEq {
        for i in 0..self.size() {
            match other.at_unchecked(i) == self.at_unchecked(i) {
                true => continue,
                false => return false,
            }
        }

        true
    }

    fn as_slice(&self) -> SetSlice<'_, Self::Item> {
        SetSlice::new(self.at_unchecked(0), 0, self.size())
    }

    fn enumerate(&self) -> SetIterator<Self::Item> {
        SetIterator::from_slice(self.as_slice())
    }
}

impl<T> Set for Vec<T> {
    type Item = T;

    fn empty() -> Self {
        vec![]
    }
    
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
    fn iter() {
        let set1 = vec![1, 2, 3, 4];
        let mut iter = set1.enumerate();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn eq() {
        let set1: Vec<u8> = vec![1, 2, 3, 4];
        let set2: Vec<u8> = vec![1, 2, 3, 4];

        assert!(set1.same(&set2));

        let set1: Vec<u8> = vec![1, 2, 3, 4];
        let set2: Vec<u8> = vec![1, 4, 8, 16];

        assert_eq!(set1.same(&set2), false);
    }
}
