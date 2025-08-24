using Printf
using DataFrames
using LinearAlgebra # Added for vec and reshape
using Base: size # Added for size(::Vector{Float64})

function riemannian_trust_region(M::AbstractManifold, cost, grad!, hess_vec_prod!, x0;
                                max_iters=100, tol=1e-8, initial_Δ=1.0)
    x = copy(x0)
    Δ = initial_Δ

    # Pre-allocate all necessary memory
    g = similar(x)

    # Setup Logging
    log_df = DataFrame(iteration=Int[], cost=Float64[], grad_norm=Float64[], delta=Float64[], rho=Float64[])

    println("--- Starting Riemannian Trust-Region Solver (Hessian-Free) ---")
    @printf("%-5s %-12s %-12s %-12s %-10s\n", "Iter", "Cost", "|g|", "Δ", "ρ")
    println("-"^55)

    for i in 1:max_iters
        f_x = cost(M, x)
        grad!(M, x, g)
        norm_g = norm(M, x, g)

        # Vectorize gradient for the subproblem solver
        g_vec = vec(g)

        # Define the Hessian-vector product operator for the subproblem solver
        # It operates on vectorized tangent vectors and returns vectorized results
        H_op(v_vec) = begin
            v = reshape(v_vec, Base.size(x)) # Reshape vectorized input to matrix
            Hv = similar(x) # Allocate Hv as matrix
            hess_vec_prod!(M, x, v, Hv) # Compute Hv as matrix
            vec(Hv) # Return vectorized Hv
        end

        # Solve the trust-region subproblem in the tangent space
        s_vec = solve_tr_subproblem(g_vec, H_op, Δ)
        s = reshape(s_vec, Base.size(x)) # Reshape solution back to original tangent vector shape

        # Evaluate step quality
        x_new = retract(M, x, s)
        f_x_new = cost(M, x_new)
        actual_reduction = f_x - f_x_new
        pred_reduction = -dot(g_vec, s_vec) - 0.5 * dot(s_vec, H_op(s_vec))
        ρ = actual_reduction / (pred_reduction + 1e-9) # Add small epsilon for stability

        @printf("%-5d %-12.4e %-12.4e %-12.4e %-10.4f\n", i, f_x, norm_g, Δ, ρ)
        push!(log_df, (iteration=i, cost=f_x, grad_norm=norm_g, delta=Δ, rho=ρ))

        if norm_g < tol
            println("\nSolver converged to tolerance in $i iterations.")
            return x, log_df
        end

        # Update point and trust radius
        if ρ > 0.1; x = x_new; end
        if ρ > 0.75 && norm(M, x, s) > 0.9 * Δ; Δ = 2.0 * Δ; elseif ρ < 0.25; Δ /= 2.0; end
    end

    println("\nSolver stopped after reaching the maximum of $max_iters iterations.")
    return x, log_df
end