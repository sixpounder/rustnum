use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::{Coord, Shape, Tensor, TensorComponent, TensorError};
use num_traits::Num;

/// Performs matrix multiplication AxB.
///
/// If A is an  m × r  matrix and B is an  r × n  matrix, then the product matrix AB is an  m × n  matrix.
///
/// If the number of columns in A doesn't match the number
/// of rows in B a `TensorError` error will be returned.
#[inline]
pub fn matmul<T: Num + Copy>(a: Tensor<T>, b: Tensor<T>) -> Result<Tensor<T>, TensorError> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a.shape().len() > 2 || b.shape().len() > 2 {
        Err(TensorError::NotSupported)
    } else {
        let target_shape: Shape;
        if a_shape[1] == b_shape[0] {
            target_shape = shape!(a_shape[0], b_shape[1]);
        } else {
            return Err(TensorError::NotSupported);
        }

        let mut out_tensor = Tensor::<T>::new_uninit(target_shape.clone());

        let op_len = target_shape[1];
        let mut row_number = 0;
        while row_number < target_shape[0] {
            let this_row: Vec<TensorComponent<T>> =
                a.iter().skip(row_number).take(op_len).collect();
            let mut col_number = 0;
            while col_number < op_len {
                let this_col: Vec<TensorComponent<T>> = b
                    .iter()
                    .skip(col_number)
                    .step_by(op_len)
                    .take(op_len)
                    .collect();
                // assert_eq!(this_row.len(), this_col.len());
                let mut acc = T::zero();
                for i in 0..op_len {
                    acc = acc + (*(this_row[i]).value * *(this_col[i]).value);
                }
                out_tensor[coord!(row_number, col_number)] = acc;
                col_number += 1;
            }
            row_number += 1;
        }

        Ok(out_tensor)
    }
}

// struct MatmulOp<'a, T> {
//     a: &'a Tensor<T>,
//     b: &'a Tensor<T>,
//     row: usize,
//     col: usize,
// }

// unsafe impl<T> Send for MatmulOp<'_, T> {}

// impl<'a, T> MatmulOp<'a, T> {
//     pub fn new(a: &'a Tensor<T>, b: &'a Tensor<T>, row: usize, col: usize) -> Self {
//         Self { a, b, row, col }
//     }
// }

struct MatmulComponent<T> {
    pub row: usize,
    pub col: usize,
    pub value: T,
}

pub fn matmul_threaded<'a, T>(a: Tensor<T>, b: Tensor<T>) -> Result<Tensor<T>, TensorError>
where
    T: Num + Copy + Send + 'static,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a.shape().len() > 2 || b.shape().len() > 2 {
        Err(TensorError::NotSupported)
    } else {
        let target_shape: Shape;
        if a_shape[1] == b_shape[0] {
            target_shape = shape!(a_shape[0], b_shape[1]);
        } else {
            return Err(TensorError::NotSupported);
        }

        let op_len = target_shape[1];
        let mut out_tensor = Tensor::<T>::new_uninit(target_shape.clone());
        let mut handles = vec![];
        let matrices = Arc::new(Mutex::new((a, b)));

        for _ in 0..target_shape.mul() {
            let local_matrices = Arc::clone(&matrices);
            let coords = coord!(0, 0);
            let t_op_len = op_len;
            let calculate = move || {
                let ab = local_matrices
                    .lock()
                    .expect("Failed to acquire lock on resource");
                let local_a = &ab.0;
                let local_b = &ab.1;
                let row = coords[0];
                let col = coords[1];
                let mut acc = T::zero();
                let this_row: Vec<TensorComponent<T>> =
                    local_a.iter().skip(row * t_op_len).take(t_op_len).collect();
                let this_col: Vec<TensorComponent<T>> = local_b
                    .iter()
                    .skip(col * t_op_len)
                    .step_by(t_op_len)
                    .take(t_op_len)
                    .collect();
                for i in 0..t_op_len {
                    acc = acc + (*(this_row[i]).value * *(this_col[i]).value);
                }
                MatmulComponent {
                    row,
                    col,
                    value: acc,
                }
            };
            let t = thread::spawn(calculate);

            handles.push(t);
        }

        for h in handles {
            let result = h.join().unwrap();
            out_tensor[coord!(result.row, result.col)] = result.value;
        }
        Ok(out_tensor)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn matmul() {
        let mut t1: Tensor<i32> = Tensor::new_uninit(shape!(2, 3));
        t1[coord!(0, 0)] = 1;
        t1[coord!(0, 1)] = 2;
        t1[coord!(0, 2)] = 4;
        t1[coord!(1, 0)] = 2;
        t1[coord!(1, 1)] = 4;
        t1[coord!(1, 2)] = 6;

        let mut t2: Tensor<i32> = Tensor::new_uninit(shape!(3, 3));
        t2[coord!(0, 0)] = 1;
        t2[coord!(0, 1)] = 2;
        t2[coord!(0, 2)] = 4;
        t2[coord!(1, 0)] = 2;
        t2[coord!(1, 1)] = 4;
        t2[coord!(1, 2)] = 6;
        t2[coord!(2, 0)] = 2;
        t2[coord!(2, 1)] = 4;
        t2[coord!(2, 2)] = 6;

        let mul_res: Result<Tensor<i32>, TensorError> = super::matmul(t1, t2);
        assert!(mul_res.is_ok());
        let t3 = mul_res.unwrap();
        assert_eq!(t3.shape(), &shape!(2, 3));
        assert_eq!(t3[coord!(0, 0)], 13); // [1, 2, 4] dot [1, 2, 2]
        assert_eq!(t3[coord!(0, 1)], 26); // [1, 2, 4] dot [2, 4, 4]
    }

    #[test]
    fn matmul_threaded() {
        let mut t1: Tensor<i32> = Tensor::new_uninit(shape!(2, 3));
        t1[coord!(0, 0)] = 1;
        t1[coord!(0, 1)] = 2;
        t1[coord!(0, 2)] = 4;
        t1[coord!(1, 0)] = 2;
        t1[coord!(1, 1)] = 4;
        t1[coord!(1, 2)] = 6;

        let mut t2: Tensor<i32> = Tensor::new_uninit(shape!(3, 3));
        t2[coord!(0, 0)] = 1;
        t2[coord!(0, 1)] = 2;
        t2[coord!(0, 2)] = 4;
        t2[coord!(1, 0)] = 2;
        t2[coord!(1, 1)] = 4;
        t2[coord!(1, 2)] = 6;
        t2[coord!(2, 0)] = 2;
        t2[coord!(2, 1)] = 4;
        t2[coord!(2, 2)] = 6;

        let mul_res: Result<Tensor<i32>, TensorError> = super::matmul_threaded(t1, t2);
        assert!(mul_res.is_ok());
        let t3 = mul_res.unwrap();
        println!("{:?}", t3);
        assert_eq!(t3.shape(), &shape!(2, 3));
        assert_eq!(t3[coord!(0, 0)], 13); // [1, 2, 4] dot [1, 2, 2]
        assert_eq!(t3[coord!(0, 1)], 26); // [1, 2, 4] dot [2, 4, 4]
    }
}
