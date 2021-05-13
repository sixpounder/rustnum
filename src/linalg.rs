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
                    .take(series_length) // Take op_len elements to much row elements
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

        let series_length = target_shape[1];
        let mut out_tensor = Tensor::<T>::new_uninit(target_shape.clone());
        let mut handles = vec![];
        let matrices = Arc::new(Mutex::new((a, b)));

        for nth_iteration in 0..target_shape[0] {
            // nth_iteration spans from 0 to the number of rows - 1 (i.e. row index)

            // Build thread local variables
            let local_matrices = Arc::clone(&matrices);
            // let coords = coord!(nth_iteration, 0);
            let row_index = nth_iteration;
            let t_series_length = series_length;
            let calculate = move || {
                // Retrieve A and B matrices
                let ab = local_matrices.lock().expect("Could not lock resource");
                    // .expect("Failed to acquire lock on resource");
                let local_a = &ab.0;
                let local_b = &ab.1;

                let mut results = vec![];

                let this_row: Vec<TensorComponent<T>> =
                    local_a.iter().skip(row_index * t_series_length).take(t_series_length).collect();

                for col_number in 0..series_length {
                    let mut acc = T::zero();

                    let this_col: Vec<TensorComponent<T>> = local_b
                        .iter()
                        .skip(col_number)
                        .step_by(t_series_length)
                        .take(t_series_length)
                        .collect();
                        for i in 0..t_series_length {
                            acc = acc + (*(this_row[i]).value * *(this_col[i]).value);
                        }

                        results.push(
                            MatmulComponent {
                                row: row_index,
                                col: col_number,
                                value: acc,
                            }
                        );
                }

                Ok(results)
            };

            let t = thread::spawn(calculate);
            handles.push(t);
        }

        for h in handles {
            let results = h.join().unwrap_or(Err(TensorError::ThreadJoin));
            let unwrapped_results = results?;
            for result in unwrapped_results.iter() {
                out_tensor[coord!(result.row, result.col)] = result.value;
            }
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
        let duration = std::time::Duration::from(end - start);
        println!("{}", duration.as_nanos());
        assert!(mul_res.is_ok());
        let t3 = mul_res.unwrap();
        assert_eq!(t3.shape(), &shape!(2, 3));
        assert_eq!(t3[coord!(0, 0)], 13); // [1, 2, 4] dot [1, 2, 2]
        assert_eq!(t3[coord!(0, 1)], 26); // [1, 2, 4] dot [2, 4, 4]
        assert_eq!(t3[coord!(1, 1)], 44); // [2, 4, 6] dot [2, 4, 4]
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

        let start = std::time::Instant::now();
        let mul_res: Result<Tensor<i32>, TensorError> = super::matmul_threaded(t1, t2);
        let end = std::time::Instant::now();
        let duration = std::time::Duration::from(end - start);
        println!("{}", duration.as_nanos());

        assert!(mul_res.is_ok());
        let t3 = mul_res.unwrap();
        assert_eq!(t3.shape(), &shape!(2, 3));
        assert_eq!(t3[coord!(0, 0)], 13); // [1, 2, 4] dot [1, 2, 2]
        assert_eq!(t3[coord!(0, 1)], 26); // [1, 2, 4] dot [2, 4, 4]
        assert_eq!(t3[coord!(1, 1)], 44); // [2, 4, 6] dot [2, 4, 4]
    }

    #[test]
    fn bench_matmul_very_big_inputs() {
        let t1: Tensor<i32> = Tensor::new(shape!(40, 40), Some(&|_, _| { 1 }));
        let t2: Tensor<i32> = Tensor::new(shape!(40, 40), Some(&|_, _| { 1 }));
        let start = std::time::Instant::now();
        let _mul_res: Result<Tensor<i32>, TensorError> = super::matmul(t1, t2);
        let end = std::time::Instant::now();
        let duration = std::time::Duration::from(end - start);
        assert!(duration.as_millis() < 3000);
    }
    #[test]
    fn bench_matmul_threaded_very_big_inputs() {
        let t1: Tensor<i32> = Tensor::new(shape!(40, 40), Some(&|_, _| { 1 }));
        let t2: Tensor<i32> = Tensor::new(shape!(40, 40), Some(&|_, _| { 1 }));
        let start = std::time::Instant::now();
        let _mul_res: Result<Tensor<i32>, TensorError> = super::matmul_threaded(t1, t2);
        let end = std::time::Instant::now();
        let duration = std::time::Duration::from(end - start);
        assert!(duration.as_millis() < 3000);
    }
}
