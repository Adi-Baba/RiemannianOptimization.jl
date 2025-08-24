# Solvers

`RiemannianOptimization.jl` provides a collection of solvers for optimization problems on Riemannian manifolds.

## Riemannian Gradient Descent

The `riemannian_gradient_descent` function performs a gradient descent optimization on a manifold.

### Usage

```julia
result = riemannian_gradient_descent(f, M, x0; step_size=0.1, max_iters=100, tol=1e-6)
```

- `f`: The cost function to minimize.
- `M`: The manifold to optimize on.
- `x0`: The initial starting point.
- `step_size`: The step size for the gradient descent.
- `max_iters`: The maximum number of iterations.
- `tol`: The gradient norm tolerance for convergence.

The function returns an `OptimizationResult` object containing the solution, the final cost, and convergence information.

## Riemannian Trust Region

The `riemannian_trust_region` function implements a trust-region method for optimization on manifolds. This is a second-order method that can converge faster than gradient descent, especially for ill-conditioned problems.

### Usage

```julia
result, log = riemannian_trust_region(M, cost, grad!, hess_vec_prod!, x0; max_iters=100, tol=1e-8, initial_Δ=1.0)
```

- `M`: The manifold.
- `cost`: The cost function.
- `grad!`: A function that computes the gradient in-place.
- `hess_vec_prod!`: A function that computes the Hessian-vector product in-place.
- `x0`: The initial starting point.
- `max_iters`: The maximum number of iterations.
- `tol`: The gradient norm tolerance.
- `initial_Δ`: The initial trust-region radius.

The function returns the solution and a log of the optimization progress.
