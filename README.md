# RiemannianOptimization.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Adi-Baba.github.io/RiemannianOptimization.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Adi-Baba.github.io/RiemannianOptimization.jl/dev)
[![Build Status](https://github.com/Adi-Baba/RiemannianOptimization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Adi-Baba/RiemannianOptimization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/Adi-Baba/RiemannianOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Adi-Baba/RiemannianOptimization.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.6%2B-purple.svg)](https://julialang.org/)

A high-performance Julia package for Riemannian optimization, providing state-of-the-art algorithms for constrained optimization on manifolds with automatic differentiation support.

## 🌟 Features

### 📐 Supported Manifolds
- **Euclidean(n)**: n-dimensional Euclidean space
- **Sphere(n)**: n-dimensional sphere embedded in Rⁿ⁺¹
- **Stiefel(n, p)**: Stiefel manifold of n×p orthogonal matrices
- **SO(3)**: Special orthogonal group for 3D rotations
- **ProductManifold**: Composition of multiple manifolds

### ⚡ Optimization Algorithms
- **Riemannian Gradient Descent**: First-order optimization with adaptive step sizes
- **Riemannian Trust-Region**: Second-order, Hessian-free method with superior convergence
- **Custom Solver Support**: Flexible interface for implementing custom algorithms

### 🧮 Automatic Differentiation
- **ScalarAD**: Simple dual numbers for basic differentiation
- **HyperDualAD**: Hyperdual numbers for exact Hessian computation
- **ChunkedAD**: High-performance chunked forward-mode AD
- **Non-allocating API**: Configuration objects for performance-critical applications

### 🚀 Performance Features
- **StaticArrays integration**: Stack-allocated arrays for maximum performance
- **Non-allocating operations**: Zero-allocation gradient and Hessian computations
- **Chunked evaluation**: Simultaneous derivative computation for efficiency
- **Precompilation**: Reduced latency for better user experience

## 📦 Installation

The package can be installed from the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/Adi-Baba/RiemannianOptimization.jl")
```

After adding the package, it's crucial to activate the project environment and instantiate its dependencies to ensure all modules are correctly loaded:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## 🚀 Quick Start

### Basic Example: Sphere Optimization



```julia
using Pkg
Pkg.activate(@__DIR__) # or Pkg.activate(".") if running from project root
Pkg.instantiate()

using RiemannianOptimization
using LinearAlgebra

# Minimize the z-coordinate on a sphere
M = Sphere(3)
f(M, p) = p[3]  # Objective function
# Initial point (normalized to lie on the sphere)
p0 = normalize([1.0, 1.0, 1.0])

# Run the Riemannian Gradient Descent optimizer
# The `verbose=true` argument prints iteration details, similar to the log you provided.
result = gradient_descent(M, f, p0; max_iter=3000)

println("\nOptimization finished.")
println("Optimal point: ", result.p)
println("Optimal value: ", result.cost)

# Expected output:
# The optimizer will converge to a point close to [0.0, 0.0, -1.0],
# which is the minimum of the z-coordinate on the 3-sphere.

## 📝 Citation

## 📊 Benchmarks

You can evaluate the performance of `RiemannianOptimization.jl` using the following benchmark suite, which leverages `BenchmarkTools.jl`.

First, ensure `BenchmarkTools` is added to your project environment:
```julia
using Pkg
Pkg.add("BenchmarkTools")
```

Now, you can copy and paste the following code into your Julia session to run the benchmarks:

```julia
using RiemannianOptimization
using BenchmarkTools
using LinearAlgebra

println("\n--- Running RiemannianOptimization Benchmarks ---")

# --- Sphere Optimization (Gradient Descent) ---
println("\nBenchmarking Sphere Optimization (Gradient Descent)...")

# Define a setup function for the benchmark to avoid including setup time in measurement
function setup_sphere_gd()
    M = Sphere(3)
    f(M, p) = p[3]
    x0 = [1/√2, 1/√2, 0.0]
    grad_f(M, p) = ScalarAD.gradient(p_ -> f(M, p_), p)
    return (M, f, grad_f, x0)
end

@benchmark gradient_descent(M, f, grad_f, x0; step_size=0.005, max_iter=10000, grad_tol=1e-6) setup=((M, f, grad_f, x0) = setup_sphere_gd()) evals=1 samples=10

# --- Euclidean Optimization (Gradient Descent) ---
println("\nBenchmarking Euclidean Optimization (Gradient Descent)...")

function setup_euclidean_gd()
    M = Euclidean(2)
    f(M, x) = x[1]^2 + x[2]^2
    x0 = [2.0, 2.0]
    grad_f(M, x) = ScalarAD.gradient(x_ -> f(M, x_), x)
    return (M, f, grad_f, x0)
end

@benchmark gradient_descent(M, f, grad_f, x0; step_size=0.1, max_iter=100, grad_tol=1e-6) setup=((M, f, grad_f, x0) = setup_euclidean_gd()) evals=1 samples=10

# --- Trust Region Optimization (using HyperDualAD for Hessian) ---
println("\nBenchmarking Trust Region Optimization (using HyperDualAD)...")

function setup_trust_region()
    M = Euclidean(2)
    f(M, x) = x[1]^4 + x[2]^4
    x0 = [2.0, 2.0]
    # Define gradient and Hessian functions using HyperDualAD
    grad!(M, x, g) = begin
        g_computed = ScalarAD.gradient(x_ -> f(M, x_), x)
        g .= g_computed
    end
    hess_vec_prod!(M, x, v, Hv) = begin
        H_computed = HyperDualAD.hessian(x_ -> f(M, x_), x)
        Hv .= H_computed * v
    end
    return (M, f, grad!, hess_vec_prod!, x0)
end

@benchmark riemannian_trust_region(M, f, grad_f, hess_f, x0) setup=((M, f, grad_f, hess_f, x0) = setup_trust_region()) evals=1 samples=10

println("\n--- Benchmarks Complete ---")
```

If you use RiemannianOptimization.jl in your research, please cite:

```bibtex
@software{RiemannianOptimization,
  author = {Aditya},
  title = {RiemannianOptimization.jl: Optimization on Manifolds in Julia},
  year = {2025},
  url = {https://github.com/Adi-Baba/RiemannianOptimization.jl}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.