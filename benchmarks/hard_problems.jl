using Pkg

# Ensure the environment uses the local, development version of the package.
# This is crucial for running tests and benchmarks against the current code.
Pkg.activate(@__DIR__)
Pkg.develop(path="..")
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
    # The Riemannian Hessian-vector product is Hess f(x)[v] = P_x(H_E f(x)[v]) - ⟨grad_E f(x), x⟩v
    # For f(x) = -xᵀLx, grad_E f(x) = -2Lx and H_E f(x)[v] = -2Lv.
    # The curvature term is -⟨-2Lx, x⟩v = 2(xᵀLx)v.
    euclidean_hess_v = -2 * (L * v)
    Hv .= project_tangent(M, x, euclidean_hess_v) # Project the Euclidean Hessian

    # Add the curvature term. Note that -xᵀLx is the cost.
    cost = max_cut_cost(M, x, L)
    Hv .+= -2 * cost .* v
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
    # The Riemannian Hessian includes a curvature term for the Stiefel manifold.
    # Hess f(X)[V] = P_X( Hess_E f(X)[V] - V * sym(Xᵀ * grad_E f(X)) )
    G_euclidean = 2 * A' * (A * X - B)
    HV_euclidean = 2 * A' * (A * V)

    # Curvature term: sym(X' * G_euclidean) * V. The order of multiplication is critical
    # for the Hessian operator to be self-adjoint.
    curvature_term = ((X' * G_euclidean) + (G_euclidean' * X)) / 2 * V

    # Project the full Euclidean term onto the tangent space
    HV .= project_tangent(M, X, HV_euclidean - curvature_term)
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
    z = reshape(p[M.indices[2]], size(M.manifolds[2]))
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

    M_sphere = M.manifolds[2]
    # Project HV_z_euclidean onto the tangent space of the sphere at z
    HV_z = project_tangent(M_sphere, z, HV_z_euclidean)

    # Add curvature term for the sphere component: -⟨∇_{E,z} f, z⟩ V_z
    # ∇_{E,z} f = -2y + 4z - 2c
    # ⟨∇_{E,z} f, z⟩ = -2yᵀz + 4zᵀz - 2cᵀz = -2yᵀz + 4 - 2cᵀz
    # We subtract this term, which is equivalent to adding (2yᵀz - 4 + 2cᵀz)V_z
    curvature_factor = 2 * dot(y, z) - 4 + 2 * dot(c, z)
    HV_z .+= curvature_factor .* V_z

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

# This allows the script to be run directly from the command line,
# e.g., `julia benchmarks/hard_problems.jl`
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_hard_benchmarks()
end

# You can also run these benchmarks interactively in a Julia REPL:
# 1. `include("benchmarks/hard_problems.jl")`
# 2. `run_all_hard_benchmarks()` or run individual benchmarks like `run_max_cut_benchmark()`.