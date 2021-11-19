**Rustum** is a small library to work with numbers and numbers distributions. All distributions support shaping
their output in custom spaces (eg. a tensor 3 x 2 x 5)

# Overview

## Structures

### Tensors

A tensor is simply a multidimensional array containing items of some kind. It is the struct
returned by many of the distribution generators in this crate.

**Create a tensor, read and write values**

```rust
let generator = &|coord: &Coord, counter: u64| {
    // For example, make every number equal to
    // the cardinality of the coordinate plus the counter
    coord.cardinality() as f64 + counter as f64
};
let mut tensor: Tensor<f64> = Tensor::new(
    shape!(3, 4, 10),
    Some(generator)
);

// Or, more simply:
let mut tensor = tensor!((2, 2, 1) => [3, 4, 2, 1])

// Get values
tensor.at(coord!(0, 0, 1));
// or
tensor[coord!(0, 0, 2)];

// Set values
tensor[coord!(0, 1, 2)] = 0.5;
// or
tensor.set(&coord!(0, 1, 2), 0.5);
```

## Numbers

### Complex numbers

```rust
let c1: Complex<f64> = Complex::new(12.0, 5.0);
assert_eq!(c1.conjugate(), Complex::new(12.0, -5.0));
```

## Distributions

### Examples

Generate a normal probability distribution:

```rust
let normal_distribution = rustnum::distributions::normal(-5.0..4.9, 0.1, 0.0, 0.2);
```

Generate an evenly spaced number range:
```rust
let range = rustnum::distributions::arange(-5.0..4.9, 0.1);
```
