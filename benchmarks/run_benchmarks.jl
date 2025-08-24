using Pkg
# Ensure the environment uses the local, development version of the package.
# This is crucial for running tests and benchmarks against the current code.
Pkg.activate(@__DIR__)
Pkg.develop(path="..")
Pkg.instantiate()

using RiemannianOptimization
using BenchmarkTools
using LinearAlgebra

# --- Setup Functions for Benchmarks ---
# Define setup functions in the global scope so @benchmark can find them.

# Setup for Sphere Optimization (Gradient Descent)
function setup_sphere_gd()
    M = Sphere(3)
    f(M, p) = p[3]
    x0 = [1/√2, 1/√2, 0.0]
    # grad_f now accepts the manifold M, consistent with the solver API.
    grad_f(M, p) = ScalarAD.gradient(p_ -> f(M, p_), p)
    return (M, f, grad_f, x0)
end

# Setup for Euclidean Optimization (Gradient Descent)
function setup_euclidean_gd()
    M = Euclidean(2)
    f(M, x) = x[1]^2 + x[2]^2
    x0 = [2.0, 2.0]
    grad_f(M, x) = ScalarAD.gradient(x_ -> f(M, x_), x)
    return (M, f, grad_f, x0)
end

# Setup for Trust Region Optimization (using HyperDualAD for Hessian)
function setup_trust_region()
    M = Euclidean(2)
    f(M, x) = x[1]^4 + x[2]^4
    x0 = [2.0, 2.0]
    # Define gradient and Hessian functions using the expected signatures
    grad!(M, x, g) = begin
        g_computed = ScalarAD.gradient(p_dual -> f(M, p_dual), x)
        g .= g_computed
    end
    # Compute the Hessian-vector product efficiently using forward-mode AD on the gradient function.
    hess_vec_prod!(M, x, v, Hv) = begin
        # H*v = d/dε [grad(f(x + εv))] at ε=0
        g_dual = ScalarAD.gradient(p -> f(M, p), RiemannianOptimization.ScalarAD.Dual.(x, v))
        Hv .= RiemannianOptimization.ScalarAD.dual.(g_dual)
    end
    return (M, f, grad!, hess_vec_prod!, x0)
end

function run_all_benchmarks()
    println("\n--- Running RiemannianOptimization Benchmarks ---")

    # --- Sphere Optimization (Gradient Descent) ---
    println("\nBenchmarking Sphere Optimization (Gradient Descent)...")
    @benchmark gradient_descent(M, f, grad_f, x0; step_size=0.005, max_iters=10000, grad_tol=1e-6) setup=((M, f, grad_f, x0) = setup_sphere_gd()) evals=1 samples=10

    # --- Euclidean Optimization (Gradient Descent) ---
    println("\nBenchmarking Euclidean Optimization (Gradient Descent)...")
    @benchmark gradient_descent(M, f, grad_f, x0; step_size=0.1, max_iters=100, grad_tol=1e-6) setup=((M, f, grad_f, x0) = setup_euclidean_gd()) evals=1 samples=10

    # --- Trust Region Optimization (using HyperDualAD for Hessian) ---
    println("\nBenchmarking Trust Region Optimization (using ScalarAD)...")
    @benchmark riemannian_trust_region(M, f, grad!, hess_vec_prod!, x0) setup=((M, f, grad!, hess_vec_prod!, x0) = setup_trust_region()) evals=1 samples=10

    println("\n--- Benchmarks Complete ---")
end

# This allows the script to be run directly from the command line,
# e.g., `julia benchmarks/run_benchmarks.jl`
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_benchmarks()
end

# You can also run these benchmarks interactively in a Julia REPL:
# 1. `include("benchmarks/run_benchmarks.jl")`
# 2. `run_all_benchmarks()`
