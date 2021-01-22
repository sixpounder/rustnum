// use crate::{Shape, Tensor, TensorError};

// pub fn dot<T: std::ops::Mul<Output = T>>(a: Tensor<T>, b: Tensor<T>) -> Result<Tensor<T>, TensorError> {
//     let a_shape = a.shape();
//     let b_shape = b.shape();
//     if a.shape().len() > 2 || b.shape().len() > 2 {
//         Err(TensorError::NotSupported)
//     } else {
//         let target_shape: Shape;
//         if a_shape[1] == b_shape[0] {
//             target_shape = shape!(a_shape[0], b_shape[1]);
//         } else {
//             return Err(TensorError::NotSupported);
//         }

//         let out_tensor = Tensor::<T>::new_uninit(target_shape);

//         for row in a.iter() {
//             let r_sum = 0.0;
//             for col in b.iter() {
//                 r_sum = r_sum + (*row.value * *col.value);
//             }
//         }

//         Ok(out_tensor)
//     }
// }
