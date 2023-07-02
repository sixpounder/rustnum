use crate::prelude::*;
use crate::{coord, Tensor, TensorComponent, TensorError, shape};
use num_traits::Num;

/// Performs matrix multiplication AxB.
///
/// If A is an  m × r  matrix and B is an  r × n  matrix, then the product matrix AB is an  m × n  matrix.
///
/// If the number of columns in A doesn't match the number
/// of rows in B a `TensorError::IncompatibleShape` error will be returned.
#[inline]
pub fn matmul<T: Num + Copy>(a: Tensor<T>, b: Tensor<T>) -> Result<Tensor<T>, TensorError> {
    let a_shape = a.shape_ref();
    let b_shape = b.shape_ref();
    if a.shape_ref().len() > 2 || b.shape_ref().len() > 2 {
        Err(TensorError::NotSupported)
    } else {
        let target_shape: Shape = if a_shape[1] == b_shape[0] {
            shape!(a_shape[0], b_shape[1])
        } else {
            return Err(TensorError::IncompatibleShape);
        };

        let mut out_tensor = Tensor::<T>::new_uninit(target_shape.clone());

        let series_length = target_shape[1];
        let mut row_number = 0;
        while row_number < target_shape[0] {
            // The components of the row
            let this_row: Vec<TensorComponent<T>> =
                a.iter().skip(row_number * series_length).take(series_length).collect();

            // Multiply for each column components
            let mut col_number = 0;
            while col_number < series_length {
                // N-th column components
                let this_col: Vec<TensorComponent<T>> = b
                    .iter()
                    .skip(col_number) // Start from the col_number-th element, to jump columns
                    .step_by(series_length) // One element each op_len, to jump rows
                    .take(series_length) // Take op_len elements to match row elements
                    .collect();

                // Run the series
                let mut acc = T::zero();
                for i in 0..series_length {
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

#[cfg(test)]
mod test {
    use crate::{tensor, TensorLike};

    use super::*;
    #[test]
    fn matmul() {
        let t1: Tensor<i32> = tensor!((2, 3) => [1, 2, 4, 2, 4, 6]);
        let t2: Tensor<i32> = tensor!((3, 3) => [1, 2, 4, 2, 4, 6, 2, 4, 6]);

        let start = std::time::Instant::now();
        let mul_res: Result<Tensor<i32>, TensorError> = super::matmul(t1, t2);
        let end = std::time::Instant::now();
        let _duration = std::time::Duration::from(end - start);

        assert!(mul_res.is_ok());
        let t3 = mul_res.unwrap();
        assert_eq!(t3.shape(), shape!(2, 3));
        assert_eq!(t3[coord!(0, 0)], 13); // [1, 2, 4] dot [1, 2, 2]
        assert_eq!(t3[coord!(0, 1)], 26); // [1, 2, 4] dot [2, 4, 4]
        assert_eq!(t3[coord!(1, 1)], 44); // [2, 4, 6] dot [2, 4, 4]
    }
}
