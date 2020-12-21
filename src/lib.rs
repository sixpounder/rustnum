//! Rustum is a small library to work with numbers and numbers distributions. All distributions support shaping
//! their output in custom spaces (eg. a tensor 3 x 2 x 5)
//! 
//! # Quick start
//! ```
//! let normal_distribution = rustnum::random::normal(3, 4, Some(shape!(3,2,5)));
//! ```
#[macro_use]
mod tensors;
mod generators;
pub mod random;

pub use tensors::*;
pub use random::*;
pub use generators::*;
