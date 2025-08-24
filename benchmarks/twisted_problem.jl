using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RiemannianOptimization
using Manifolds
using LinearAlgebra
using Printf

# Define the Rosenbrock function
function rosenbrock_cost(M::RiemannianOptimization.Euclidean, x)
    return (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
end

# Define the gradient of the Rosenbrock function
function rosenbrock_grad!(M::RiemannianOptimization.Euclidean, x, g)
    g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
    return g
end

# Define the Hessian-vector product of the Rosenbrock function
function rosenbrock_hess_vec_prod!(M::RiemannianOptimization.Euclidean, x, v, Hv)
    Hv[1] = (2 - 400 * x[2] + 1200 * x[1]^2) * v[1] - 400 * x[1] * v[2]
    Hv[2] = -400 * x[1] * v[1] + 200 * v[2]
    return Hv
end

function run_twisted_problem_benchmark()
    M = RiemannianOptimization.Euclidean(2) # 2-dimensional Euclidean manifold
    x0 = [-1.2, 1.0] # Starting point for Rosenbrock function

    println("--- Starting Riemannian Trust-Region Solver on Rosenbrock Function ---")
    x_min, log_df = riemannian_trust_region(M, rosenbrock_cost, rosenbrock_grad!, rosenbrock_hess_vec_prod!, x0)

    println("\nOptimization complete.")
    println("Minimum found at: ", x_min)
    println("Cost at minimum: ", rosenbrock_cost(M, x_min))
    println("Gradient norm at minimum: ", RiemannianOptimization.norm(M, x_min, rosenbrock_grad!(M, x_min, similar(x_min))))

    println("\n--- Optimization Log ---")
    println(log_df)
end

# To run this benchmark:
# 1. Start Julia in your project directory.
# 2. Activate the project: `pkg> activate .`
# 3. Include this file: `include("benchmarks/twisted_problem.jl")`
# 4. Run the benchmark function: `run_twisted_problem_benchmark()`