using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RiemannianOptimization
using LinearAlgebra

# 1. Define the Sphere manifold
M = Sphere(3)

# 2. Define the cost function
# We want to find the point on the sphere with the smallest z-coordinate.
# This is equivalent to minimizing f(p) = p[3].
f(p) = p[3]

# 3. Choose a starting point
# A point on the equator of the sphere
x0 = [1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0]

# 4. Run the optimization
# We expect the solution to be the south pole [0, 0, -1]
result = gradient_descent(f, M, x0, step_size=0.1, max_iters=50, grad_tol=1e-6)

# 5. Print the result
println("Converged: ", result.converged)
println("Solution: ", result.solution)
println("Expected solution: [0.0, 0.0, -1.0]")
println("Final cost: ", result.cost)