"""
    GradFunctor

A callable struct (functor) used to wrap the gradient computation.
This is a more robust alternative to a closure, as it can prevent
allocations associated with capturing variables.
"""
struct GradFunctor{F, Cfg}
    f::F
    cfg::Cfg
end

function (gf::GradFunctor)(x_vec)
    gradient!(gf.cfg.grad_buffer, gf.f, x_vec, gf.cfg.grad_cfg)
    return gf.cfg.grad_buffer
end

"""
    HessianConfig(X; chunk_size=8)

A configuration object that holds all pre-allocated workspaces for the
Forward-over-Forward `hessian!` calculation. Reusing this object in a
loop avoids all allocations.
"""
struct HessianConfig{N, T, L}
    grad_cfg::ChunkedAD.GradientConfig{N, ChunkedAD.MultiDual{N, T}}
    grad_buffer::Vector{ChunkedAD.MultiDual{N, T}}
    jac_cfg::ChunkedAD.JacobianConfig{N, T, L}
end

function HessianConfig(X::AbstractVector{T}; chunk_size=8) where T
    N = chunk_size
    L = length(X)
    nested_dual_type = ChunkedAD.MultiDual{N, T}

    grad_cfg_template = similar(X, nested_dual_type)
    grad_cfg = ChunkedAD.GradientConfig(grad_cfg_template; chunk_size=N)
    grad_buffer = similar(grad_cfg_template)
    
    jac_cfg = ChunkedAD.JacobianConfig(L, X; chunk_size=N)

    return HessianConfig{N, T, L}(grad_cfg, grad_buffer, jac_cfg)
end

"""
    hessian!(H, f, X, cfg::HessianConfig)

The core, non-allocating hessian computation function that uses a pre-configured
workspace. This is the version that should be used inside a solver loop.
"""
function hessian!(H, f, X::AbstractVector, cfg::HessianConfig{N, T, L}) where {N, T, L}
    # Create an instance of our non-allocating functor
    g = GradFunctor(f, cfg)

    # Compute the Jacobian of the gradient functor `g`
    jacobian!(H, g, X, cfg.jac_cfg)

    # The result from AD is not guaranteed to be perfectly symmetric due to
    # floating-point arithmetic. Enforce symmetry in a non-allocating way
    inds = axes(H, 1)
    @inbounds for i in inds
        for j in (i+1):last(inds)
            H[i, j] = H[j, i] = (H[i, j] + H[j, i]) / 2
        end
    end
    return H
end

"""
    hessian!(H, f, X; chunk_size=8)

Computes the Hessian of `f` at `X` by taking the Jacobian of the gradient.
This is a highly efficient "Forward-over-Forward" implementation that reuses
the chunked MultiDual engine.
"""
function hessian!(H, f, X::AbstractVector{T}; chunk_size=8) where T
    # This is now a convenience wrapper that creates the config on the fly
    cfg = HessianConfig(X; chunk_size=chunk_size)
    hessian!(H, f, X, cfg)
    return H
end

"""
    hessian(f, X; chunk_size=8)

User-facing, allocating version of `hessian!`.
"""
function hessian(f, X::AbstractVector{T}; chunk_size=8) where T
    H = similar(X, length(X), length(X))
    hessian!(H, f, X; chunk_size=chunk_size)
    return H
end