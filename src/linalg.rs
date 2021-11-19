use crate::{Coord, coord, Shape, Tensor, TensorComponent, TensorError, shape};
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
        let target_shape: Shape;
        if a_shape[1] == b_shape[0] {
            target_shape = shape!(a_shape[0], b_shape[1]);
        } else {
            return Err(TensorError::IncompatibleShape);
        }

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

        let start = std::time::Instant::now();
        let mul_res: Result<Tensor<i32>, TensorError> = super::matmul(t1, t2);
        let end = std::time::Instant::now();
        let _duration = std::time::Duration::from(end - start);
        // println!("{}", duration.as_nanos());
        assert!(mul_res.is_ok());
        let t3 = mul_res.unwrap();
        assert_eq!(t3.shape_ref(), &shape!(2, 3));
        assert_eq!(t3[coord!(0, 0)], 13); // [1, 2, 4] dot [1, 2, 2]
        assert_eq!(t3[coord!(0, 1)], 26); // [1, 2, 4] dot [2, 4, 4]
        assert_eq!(t3[coord!(1, 1)], 44); // [2, 4, 6] dot [2, 4, 4]
    }
}
