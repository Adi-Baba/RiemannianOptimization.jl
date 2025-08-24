# Automatic Differentiation

The `RiemannianOptimization.jl` package provides a powerful and flexible automatic differentiation (AD) system tailored for optimization on manifolds. It features several backends that can be used to compute gradients and Hessians efficiently.

## ScalarAD

`ScalarAD` is a simple and lightweight AD engine based on dual numbers. It is suitable for computing first-order derivatives of scalar functions.

### Usage

To compute the derivative of a function `f` at a point `x`, you can use the `derivative` function:

```julia
using RiemannianOptimization.ScalarAD

f(x) = x^2
deriv = derivative(f, 2.0)  # Returns 4.0
```

To compute the gradient of a function that takes a vector input, you can use the `gradient` function:

```julia
using RiemannianOptimization.ScalarAD

g(X) = X[1]^2 + X[2]^3
X = [1.0, 2.0]
grad = gradient(g, X)  # Returns [2.0, 12.0]
```

## HyperDualAD

`HyperDualAD` uses hyper-dual numbers to compute exact second-order derivatives (Hessians) of scalar functions.

### Usage

To compute the second derivative of a function `f` at a point `x`:

```julia
using RiemannianOptimization.HyperDualAD

f(x) = x^3
h_ssian = second_derivative(f, 2.0)  # Returns 12.0
```

To compute the gradient and Hessian of a vector-input function:

```julia
using RiemannianOptimization.HyperDualAD

g(X) = X[1]^2 + X[2]^3
X = [1.0, 2.0]
grad, H = gradient_hessian(g, X)
```

## ChunkedAD

`ChunkedAD` is a high-performance AD engine that uses chunked forward-mode automatic differentiation. It is designed for efficiency and can compute gradients and Jacobians of vector-valued functions with minimal allocations.

### Usage

To compute the gradient of a function:

```julia
using RiemannianOptimization.ChunkedAD

g(X) = X[1]^2 + X[2]^3
X = [1.0, 2.0]
grad = gradient(g, X)
```

To compute the Jacobian of a vector-valued function:

```julia
using RiemannianOptimization.ChunkedAD

h(X) = [X[1]^2, X[2]^3]
X = [1.0, 2.0]
J = jacobian(h, X)
```
