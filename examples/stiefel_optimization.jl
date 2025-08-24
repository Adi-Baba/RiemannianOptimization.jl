using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RiemannianOptimization
using LinearAlgebra

# 1. Define the Stiefel manifold
# We will work with 3x2 orthonormal matrices
M = Stiefel(3, 2)

# 2. Define the cost function
# We want to find the Stiefel matrix X that is closest to a target matrix A.
# This is equivalent to minimizing f(X) = ||X - A||^2.
A = randn(3, 2)
f(X) = norm(X - A)^2

# 3. Choose a starting point
# A random point on the Stiefel manifold
x0 = project(M, randn(3, 2))

# 4. Run the optimization
result = gradient_descent(f, M, x0, step_size=0.01, max_iters=100, grad_tol=1e-6)

# 5. Print the result
println("Converged: ", result.converged)
println("Solution:\n", result.solution)
println("Final cost: ", result.cost)

# The optimal solution is the polar decomposition of A
U, S, V = svd(A)
println("Expected solution:\n", U * V')