//! Rustum is a small library to work with numbers and numbers distributions. All distributions support shaping
//! their output in custom spaces (eg. a tensor 3 x 2 x 5)
//! 
//! # Quick start
//! ```
//! let normal_distribution = rustnum::distributions::normal(-5.0..4.9, 0.1, 0.0, 0.2);
//! ```
#[macro_use]
mod tensors;
mod generators;
pub mod distributions;
pub mod range;

pub use tensors::*;
pub use generators::*;
