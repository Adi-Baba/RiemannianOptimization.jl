using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RiemannianOptimization
using LinearAlgebra

# The Procrustes problem is about finding a rotation matrix R that best aligns
# two sets of points, A and B, by minimizing ||RA - B||^2.

# 1. Generate some data
# Let's create a known rotation and then try to recover it.
n = 3
p = 10
R_true = project(SO3(), randn(n, n))
A = randn(n, p)
B = R_true * A + 0.1 * randn(n, p)  # Add some noise

# 2. Define the manifold
# We are looking for a rotation matrix, which is an element of SO(3).
M = SO3()

# 3. Define the cost function
# We want to minimize f(R) = ||RA - B||^2
f(R) = LinearAlgebra.norm(R * A - B)^2

# Riemannian gradient of f(R) on SO(3)
grad_f(R) = project_tangent(M, R, 2 * (R * A - B) * A')

# 4. Choose a starting point
# A random rotation matrix
R0 = project(M, randn(n, n))

# 5. Run the optimization
result = gradient_descent(M, f, grad_f, R0, step_size=0.01, max_iters=100, grad_tol=1e-6)

# 6. Print the result
println("Converged: ", result.converged)
println("Found rotation:\n", result.solution)
println("True rotation:\n", R_true)
println("Final cost: ", result.cost)