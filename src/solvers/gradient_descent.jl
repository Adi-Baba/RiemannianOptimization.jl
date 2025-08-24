

"""
    gradient_descent(M::AbstractManifold, f, grad_f, p_initial; kwargs...)

Perform a gradient descent optimization on the manifold `M`.

# Arguments
- `M`: The manifold to optimize on.
- `f`: The cost function `f(M, p)`.
- `grad_f`: The gradient of the cost function `grad_f(M, p)`.
- `p_initial`: The starting point for the optimization.

# Keyword Arguments
- `max_iters::Int=100`: Maximum number of iterations.
- `step_size::Float64=0.01`: Initial step size for the line search.
- `grad_tol::Float64=1e-6`: Gradient norm tolerance for stopping.
"""
function gradient_descent(
    M::AbstractManifold,
    f,      # f(M, p)
    grad_f, # grad_f(M, p)
    p_initial;
    max_iters::Int=100,
    step_size::Float64=0.01,
    grad_tol::Float64=1e-6
)
    p = copy(p_initial)
    
    for i in 1:max_iters
        # Compute gradient at the current point
        # This check provides backward compatibility for functions with signature grad_f(p).
        g = if applicable(grad_f, M, p)
            grad_f(M, p)
        elseif applicable(grad_f, p)
            @warn "The signature `grad_f(p)` is deprecated. Please use `grad_f(M, p)`." maxlog=1
            grad_f(p)
        else
            error("The provided gradient function `grad_f` has an unsupported signature. Expected `grad_f(M, p)` or `grad_f(p)`.")
        end
        
        # Project gradient onto the tangent space (ensures correctness)
        g_proj = project_tangent(M, p, g)
        
        grad_norm = norm(M, p, g_proj)
        
        # For debugging: println("Iteration: ", i, ", grad_norm: ", grad_norm)
        if grad_norm < grad_tol
            cost = applicable(f, M, p) ? f(M, p) : f(p)
            return OptimizationResult(p, cost, true, "Gradient norm tolerance reached.")
        end
        
        # Descent direction
        descent_dir = -g_proj
        
        # Perform a simple retraction (line search could be added here)
        p_next = retract(M, p, step_size * descent_dir)
        
        # Update the point
        p = p_next
    end
    
    cost = applicable(f, M, p) ? f(M, p) : f(p)
    return OptimizationResult(p, cost, false, "Maximum iterations reached.")
end