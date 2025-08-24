using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RiemannianOptimization
using BenchmarkTools
using LinearAlgebra

println("\n--- Running RiemannianOptimization Benchmarks ---")

# --- Sphere Optimization (Gradient Descent) ---
println("\nBenchmarking Sphere Optimization (Gradient Descent)...")

# Define a setup function for the benchmark to avoid including setup time in measurement
function setup_sphere_gd()
    M = Sphere(3)
    f(M, p) = p[3] # Modified to accept M
    x0 = [1/√2, 1/√2, 0.0]
    # Use a slightly higher tolerance for benchmarks if exact convergence is not critical
    # or if you want to see performance for reaching a 'good enough' solution.
    # For robust benchmarks, ensure max_iters is high enough for convergence.
    return (M, p_arg -> f(M, p_arg), p_val -> ScalarAD.gradient(p_dual -> f(M, p_dual), p_val), x0)
end

@benchmark gradient_descent(M, f, grad_f, x0; step_size=0.005, max_iters=10000, grad_tol=1e-6) setup=((M, f, grad_f, x0) = setup_sphere_gd()) evals=1 samples=10

# --- Euclidean Optimization (Gradient Descent) ---
println("\nBenchmarking Euclidean Optimization (Gradient Descent)...")

function setup_euclidean_gd()
    M = Euclidean(2)
    f(M, x) = x[1]^2 + x[2]^2 # Modified to accept M
    x0 = [2.0, 2.0]
    return (M, p_arg -> f(M, p_arg), p_val -> ScalarAD.gradient(p_dual -> f(M, p_dual), p_val), x0)
end

@benchmark gradient_descent(M, f, grad_f, x0; step_size=0.1, max_iters=100, grad_tol=1e-6) setup=((M, f, grad_f, x0) = setup_euclidean_gd()) evals=1 samples=10

# --- Trust Region Optimization (using HyperDualAD for Hessian) ---
println("\nBenchmarking Trust Region Optimization (using HyperDualAD)...")

function setup_trust_region()
    M = Euclidean(2)
    f(M, x) = x[1]^4 + x[2]^4 # Modified to accept M
    x0 = [2.0, 2.0]
    # Define gradient and Hessian functions using HyperDualAD
    grad!(M, x, g) = begin
        g_computed = ScalarAD.gradient(p_dual -> f(M, p_dual), x)
        g .= g_computed
    end
    hess_vec_prod!(M, x, v, Hv) = begin
        H_computed = HyperDualAD.hessian(p_dual -> f(M, p_dual), x)
        Hv .= H_computed * v
    end
    return (M, f, grad!, hess_vec_prod!, x0)
end

@benchmark riemannian_trust_region(M, f, grad_f, hess_f, x0) setup=((M, f, grad_f, hess_f, x0) = setup_trust_region()) evals=1 samples=10

println("\n--- Benchmarks Complete ---")
println("To run these benchmarks, navigate to the project root in Julia Pkg mode (press `]`
), then run:")
println("activate .")
println("include(\"benchmarks/run_benchmarks.jl\")")
println("\nConsider adjusting `samples` and `evals` for more precise measurements.")
println("For comparing with other libraries, you would need to integrate them into similar benchmark setups.")
