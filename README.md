# RiemannianOptimization.jl

A Julia package for performing optimization on Riemannian manifolds. It provides a flexible framework for defining manifolds, cost functions, and using advanced solvers like the Riemannian Trust-Region method.

This library is designed to be both educational and performant, with clear implementations of core concepts in Riemannian geometry and optimization.

## Features

- **Type-Dispatch Manifold System**: Easily define and work with different manifolds.
- **High-Performance Solvers**: Includes Gradient Descent and a Hessian-free Trust-Region solver.
- **Common Manifolds**: Built-in support for `Euclidean`, `Sphere`, `Stiefel`, and `ProductManifold`.
- **Automatic Differentiation**: Comes with a simple, dependency-free forward-mode AD engine (`ScalarAD.jl`) for computing gradients and Hessian-vector products.
- **Benchmark Suite**: Includes examples for standard optimization problems like Rosenbrock, Max-Cut, and Orthogonal Procrustes.

## Installation

Since this is a local package, you can add it to your Julia environment by specifying its path. From the Julia REPL, enter Pkg mode by pressing `]` and then run:

```julia
pkg> using Pkg
pkg> Pkg.add(url="https://github.com/Adi-Baba/RiemannianOptimization.jl")
```

## Quick Start: Optimizing the Rosenbrock Function

Here is a complete example of how to use the Riemannian Trust-Region solver to find the minimum of the 2D Rosenbrock function on the Euclidean manifold.

```julia
using RiemannianOptimization

# 1. Define the manifold
M = Euclidean(2)

# 2. Define the cost function and its derivatives
function rosenbrock_cost(M::Euclidean, x)
    return (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
end

function rosenbrock_grad!(M::Euclidean, x, g)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    return g
end

function rosenbrock_hess_vec_prod!(M::Euclidean, x, v, Hv)
    Hv[1] = (2 - 400 * x[2] + 1200 * x[1]^2) * v[1] - 400 * x[1] * v[2]
    Hv[2] = -400 * x[1] * v[1] + 200 * v[2]
    return Hv
end

# 3. Set an initial point and run the solver
x0 = [-1.2, 1.0]
x_min, log_df = riemannian_trust_region(
    M,
    rosenbrock_cost,
    rosenbrock_grad!,
    rosenbrock_hess_vec_prod!,
    x0
)

# 4. Print the results
println("\nOptimization complete.")
println("Minimum found at: ", x_min)
println("Cost at minimum: ", rosenbrock_cost(M, x_min))
```

### Expected Output

The solver will print its progress and converge to the known minimum at `[1.0, 1.0]`.

```
--- Starting Riemannian Trust-Region Solver (Hessian-Free) ---
Iter  Cost         |g|          Δ            ρ
-------------------------------------------------------
1     2.4200e+01   2.3287e+02   1.0000e+00   1.0028
2     4.7319e+00   4.6394e+00   1.0000e+00   -0.4143
...
24    1.0611e-13   3.3800e-07   2.5000e-01   0.0001

Solver converged: step size (3.1851e-13) below tolerance (1e-12) in 24 iterations.

Optimization complete.
Minimum found at: [0.9999996743780089, 0.9999993478371609]
Cost at minimum: 1.0611413038132374e-13
```

## Advanced Usage: Optimization on the Sphere

This library can solve problems on non-Euclidean manifolds. For example, the Max-Cut problem can be formulated as maximizing `x'Lx` subject to `x` being on the n-sphere.

To solve this, you would:
1.  Define the `Sphere` manifold: `M = Sphere(n-1)`.
2.  Implement the cost, gradient, and Hessian-vector product specific to the Sphere manifold, using `project_tangent` to ensure vectors lie in the correct tangent space.
3.  Call the solver as before.

You can find a full implementation of this and other complex problems in the `benchmarks/` directory.

## Available Components

### Solvers

- `riemannian_trust_region`: A robust, second-order method for finding local minima. Requires a Hessian-vector product.
- `gradient_descent`: A simple, first-order method. Requires only the gradient.

### Manifolds

- `Euclidean(n)`: Standard n-dimensional Euclidean space.
- `Sphere(n)`: The n-dimensional sphere embedded in `R^(n+1)`.
- `Stiefel(n, p)`: The Stiefel manifold of `p` orthonormal frames in `R^n`.
- `ProductManifold(M1, M2, ...)`: A manifold constructed as the Cartesian product of other manifolds.

## Running Benchmarks

The project includes a set of benchmarks to test solver performance on various problems. To run them, navigate to the `benchmarks/` directory and execute the scripts with Julia.

```bash
# From the root directory of the project
cd benchmarks

# Run the Rosenbrock benchmark
julia twisted_problem.jl

# Run the Max-Cut, Procrustes, and Product Manifold benchmarks
julia hard_problems.jl
```

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or suggest a feature. If you would like to contribute code, please open a pull request.

## License

This project is licensed under the MIT License.
