# RiemannianOptimization.jl

[![](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

A Julia package for optimization on Riemannian manifolds, featuring state-of-the-art first and second-order optimization algorithms with automatic differentiation support.

## Why RiemannianOptimization.jl?

This package is designed from the ground up for **performance** and **flexibility**.

- **High Performance**: All critical geometric operations (retractions, vector transports) are implemented in a non-allocating fashion, making the solvers suitable for performance-critical applications.
- **Modern AD Support**: Our custom-built automatic differentiation backend is tailored for Riemannian operations, providing accurate gradients and Hessians efficiently.
- **Clean API**: The interface is designed to be simple, extensible, and easy to integrate into larger projects.

## Features

- **Manifold Support**: Sphere, Stiefel, SO(3), Euclidean, and Product manifolds
- **Optimization Algorithms**: Riemannian gradient descent and trust-region methods
- **Automatic Differentiation**: Custom dual, hyperdual, and multidual number implementations

## Installation

```julia
using Pkg
Pkg.add("RiemannianOptimization")