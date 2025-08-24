

"""
    gradient_descent(M::AbstractManifold, f, grad_f, p_initial; kwargs...)

Perform a gradient descent optimization on the manifold `M`.

# Arguments
- `M`: The manifold to optimize on.
- `f`: The cost function `f(p)`.
- `grad_f`: The gradient of the cost function `grad_f(p)`.
- `p_initial`: The starting point for the optimization.

# Keyword Arguments
- `max_iters::Int=100`: Maximum number of iterations.
- `step_size::Float64=0.01`: Initial step size for the line search.
- `grad_tol::Float64=1e-6`: Gradient norm tolerance for stopping.
"""
function gradient_descent(
    M::AbstractManifold,
    f,
    grad_f,
    p_initial;
    max_iters::Int=100,
    step_size::Float64=0.01,
    grad_tol::Float64=1e-6
)
    p = copy(p_initial)
    
    for i in 1:max_iters
        # Compute gradient at the current point
        g = grad_f(p)
        
        # Project gradient onto the tangent space (ensures correctness)
        g_proj = project_tangent(M, p, g)
        
        grad_norm = norm(M, p, g_proj)
        
        println("Iteration: ", i, ", grad_norm: ", grad_norm, ", p: ", p, ", f(p): ", f(p))
        if grad_norm < grad_tol
            return OptimizationResult(p, f(p), true, "Gradient norm tolerance reached.")
        end
        
        # Descent direction
        descent_dir = -g_proj
        
        # Perform a simple retraction (line search could be added here)
        p_next = retract(M, p, step_size * descent_dir)
        
        # Update the point
        p = p_next
    end
    
    return OptimizationResult(p, f(p), false, "Maximum iterations reached.")
end