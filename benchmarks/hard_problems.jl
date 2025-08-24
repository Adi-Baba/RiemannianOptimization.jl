Pkg.activate(@__DIR__)
Pkg.instantiate()

using RiemannianOptimization
using Manifolds
using LinearAlgebra
using Printf
using SparseArrays # For adjacency matrix

# --- Problem 1: Max-Cut on the Sphere Manifold ---
# This is a quadratic optimization problem on the sphere.
# Maximize xᵀLx subject to xᵀx = 1, where L is the graph Laplacian.
# We minimize -xᵀLx.

function max_cut_cost(M::RiemannianOptimization.Sphere, x, L)
    return -dot(x, L, x)
end

function max_cut_grad!(M::RiemannianOptimization.Sphere, x, g, L)
    # Gradient of -xᵀLx is -2Lx.
    # Project onto the tangent space of the sphere.
    g .= -2 * (L * x)
    g .= project_tangent(M, x, g) # Project g onto the tangent space at x
    return g
end

function max_cut_hess_vec_prod!(M::RiemannianOptimization.Sphere, x, v, Hv, L)
    # Hessian of -xᵀLx is -2L.
    # The Riemannian Hessian-vector product is P_x(-2L v), where P_x is the projection onto the tangent space.
    Hv .= -2 * (L * v)
    Hv .= project_tangent(M, x, Hv) # Project Hv onto the tangent space at x
    return Hv
end

function run_max_cut_benchmark()
    println("\n--- Starting Max-Cut on Sphere Manifold Benchmark ---")

    # Define a graph (e.g., a simple cycle graph)
    num_nodes = 20
    A = spzeros(num_nodes, num_nodes)
    for i in 1:num_nodes
        A[i, mod1(i + 1, num_nodes)] = 1
        A[mod1(i + 1, num_nodes), i] = 1
    end
    D = Diagonal(vec(sum(A, dims=2)))
    L = D - A # Graph Laplacian

    M = RiemannianOptimization.Sphere(num_nodes - 1) # Sphere embedded in R^num_nodes
    x0 = normalize(randn(num_nodes)) # Random initial point on the sphere

    println("Running Riemannian Trust-Region Solver for Max-Cut...")
    x_min, log_df = riemannian_trust_region(M, (M_arg, x_arg) -> max_cut_cost(M_arg, x_arg, L), (M_in, x_in, g_out) -> max_cut_grad!(M_in, x_in, g_out, L), (M_in, x_in, v_in, Hv_out) -> max_cut_hess_vec_prod!(M_in, x_in, v_in, Hv_out, L), x0)

    println("\nOptimization complete.")
    println("Minimum found (x): ", x_min)
    println("Cost at minimum: ", max_cut_cost(M, x_min, L))
    println("Gradient norm at minimum: ", RiemannianOptimization.norm(M, x_min, max_cut_grad!(M, x_min, similar(x_min), L)))

    println("\n--- Optimization Log ---")
    println(log_df)
end

# --- Problem 2: Orthogonal Procrustes on the Stiefel Manifold ---
# Minimize ||AX - B||_F^2 subject to XᵀX = I, where X is on the Stiefel manifold.
# This is a classic problem, but can be made "harder" by increasing dimensions or adding noise.

function procrustes_cost(M::RiemannianOptimization.Stiefel, X, A, B)
    return LinearAlgebra.norm(A * X - B)^2
end

function procrustes_grad!(M::RiemannianOptimization.Stiefel, X, G, A, B)
    # Euclidean gradient of ||AX - B||_F^2 is 2Aᵀ(AX - B)
    G_euclidean = 2 * A' * (A * X - B)
    G .= project_tangent(M, X, G_euclidean) # Project onto the tangent space of the Stiefel manifold
    return G
end

function procrustes_hess_vec_prod!(M::RiemannianOptimization.Stiefel, X, V, HV, A, B)
    # Euclidean Hessian of ||AX - B||_F^2 is 2AᵀA.
    # Riemannian Hessian-vector product involves the Euclidean Hessian projected.
    HV_euclidean = 2 * A' * (A * V)
    HV .= project_tangent(M, X, HV_euclidean) # Project onto the tangent space of the Stiefel manifold
    return HV
end

function run_procrustes_benchmark()
    println("\n--- Starting Orthogonal Procrustes on Stiefel Manifold Benchmark ---")

    m, n = 50, 10 # A is m x n, X is n x n, B is m x n
    A = randn(m, n)
    B = randn(m, n)

    M = RiemannianOptimization.Stiefel(n, n) # Stiefel manifold of n-frames in R^n
    X0 = Matrix(qr(randn(n, n)).Q) # Random initial orthogonal matrix

    println("Running Riemannian Trust-Region Solver for Orthogonal Procrustes...")
    X_min, log_df = riemannian_trust_region(M, (M_arg, X_arg) -> procrustes_cost(M_arg, X_arg, A, B), (M_in, X_in, G_out) -> procrustes_grad!(M_in, X_in, G_out, A, B), (M_in, X_in, V_in, HV_out) -> procrustes_hess_vec_prod!(M_in, X_in, V_in, HV_out, A, B), X0)

    println("\nOptimization complete.")
    println("Minimum found (X): ", X_min)
    println("Cost at minimum: ", procrustes_cost(M, X_min, A, B))
    println("Gradient norm at minimum: ", RiemannianOptimization.norm(M, X_min, procrustes_grad!(M, X_min, similar(X_min), A, B)))

    println("\n--- Optimization Log ---")
    println(log_df)
end

# --- Problem 3: Combined Euclidean and Sphere Optimization ---
# Optimize a function that depends on both a Euclidean vector and a point on a sphere.
# Example: Minimize f(y, z) = ||y - z||^2 + ||y||^2 + ||z - c||^2
# where y is Euclidean and z is on the Sphere.

function combined_cost(M::RiemannianOptimization.ProductManifold, p::AbstractVector, c)
    y = p[M.indices[1]]
    z = reshape(p[M.indices[2]], Base.size(M.manifolds[2]))
    return LinearAlgebra.norm(y - z)^2 + LinearAlgebra.norm(y)^2 + LinearAlgebra.norm(z - c)^2
end

function combined_grad!(M::RiemannianOptimization.ProductManifold, p::AbstractVector, G::AbstractVector, c)
    y = p[M.indices[1]]
    z = reshape(p[M.indices[2]], size(M.manifolds[2]))

    # Euclidean gradient for y: 2(y - z) + 2y = 4y - 2z
    G_y = 4 * y - 2 * z

    # Euclidean gradient for z: -2(y - z) + 2(z - c) = -2y + 4z - 2c
    G_z_euclidean = -2 * y + 4 * z - 2 * c

    # Project G_z_euclidean onto the tangent space of the sphere at z
    G_z = project_tangent(M.manifolds[2], z, G_z_euclidean)

    G[M.indices[1]] .= G_y
    G[M.indices[2]] .= vec(G_z)
    return G
end

function combined_hess_vec_prod!(M::RiemannianOptimization.ProductManifold, p::AbstractVector, V::AbstractVector, HV::AbstractVector, c)
    y = p[M.indices[1]]
    z = reshape(p[M.indices[2]], size(M.manifolds[2]))
    V_y = V[M.indices[1]]
    V_z = reshape(V[M.indices[2]], size(M.manifolds[2]))

    # Euclidean Hessian-vector product for y: 4V_y - 2V_z
    HV_y = 4 * V_y - 2 * V_z

    # Euclidean Hessian-vector product for z: -2V_y + 4V_z
    HV_z_euclidean = -2 * V_y + 4 * V_z

    # Project HV_z_euclidean onto the tangent space of the sphere at z
    HV_z = project_tangent(M.manifolds[2], z, HV_z_euclidean)

    HV[M.indices[1]] .= HV_y
    HV[M.indices[2]] .= vec(HV_z)
    return HV
end

function run_combined_benchmark()
    println("\n--- Starting Combined Euclidean and Sphere Optimization Benchmark ---")

    dim_euclidean = 5
    dim_sphere = 5
    
    M_euclidean = RiemannianOptimization.Euclidean(dim_euclidean)
    M_sphere = RiemannianOptimization.Sphere(dim_sphere - 1) # Sphere embedded in R^dim_sphere
    M = RiemannianOptimization.ProductManifold(M_euclidean, M_sphere)

    # Initial point
    y0 = randn(dim_euclidean)
    z0 = normalize(randn(dim_sphere))
    p0 = [y0; z0]

    # Target point for z (on the sphere)
    c = normalize(randn(dim_sphere))

    println("Running Riemannian Trust-Region Solver for Combined Optimization...")
    p_min, log_df = riemannian_trust_region(M, (M_arg, p_arg) -> combined_cost(M_arg, p_arg, c), (M_arg, p_arg, G_arg) -> combined_grad!(M_arg, p_arg, G_arg, c), (M_arg, p_arg, V_arg, HV_arg) -> combined_hess_vec_prod!(M_arg, p_arg, V_arg, HV_arg, c), p0)

    println("\nOptimization complete.")
    p_min_y = p_min[M.indices[1]]
    p_min_z = reshape(p_min[M.indices[2]], size(M.manifolds[2]))
    println("Minimum found (y): ", p_min_y)
    println("Minimum found (z): ", p_min_z)
    println("Cost at minimum: ", combined_cost(M, p_min, c))
    g_min = similar(p0)
    combined_grad!(M, p_min, g_min, c)
    println("Gradient norm at minimum: ", RiemannianOptimization.norm(M, p_min, g_min))

    println("\n--- Optimization Log ---")
    println(log_df)
end


function run_all_hard_benchmarks()
    run_max_cut_benchmark()
    run_procrustes_benchmark()
    run_combined_benchmark()
end

# To run these benchmarks:
# 1. Start Julia in your project directory.
# 2. Activate the project: `pkg> activate .`
# 3. Include this file: `include("benchmarks/hard_problems.jl")`
# 4. Run all benchmarks: `run_all_hard_benchmarks()`
#    Or run individual benchmarks: `run_max_cut_benchmark()`, `run_procrustes_benchmark()`, `run_combined_benchmark()`