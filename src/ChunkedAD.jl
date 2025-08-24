# ======================================================================
# MultiDual Chunked Automatic Differentiation
# Author: Aditya
# Version: 2.0 (2025)
#
# Features:
#   - Chunked forward-mode AD for massive speedups in gradient/Jacobian computation.
#   - Non-allocating core functions for use in performance-critical loops.
#   - Configuration objects (`GradientConfig`, `JacobianConfig`) for easy memory management.
#   - Built on StaticArrays for optimal performance.
# ======================================================================

module ChunkedAD

export MultiDual, gradient, jacobian, gradient!, jacobian!, loss_and_gradient!
export GradientConfig, JacobianConfig

# We need these dependencies inside the module
using StaticArrays
using LinearAlgebra # For the identity matrix `I`

# Include the scalar AD implementation as a submodule.
# This makes it available as ChunkedAD.ScalarAD for testing purposes.
# The ScalarAD.jl file should be in the same `src` directory.
include("ScalarAD.jl")

"""
    MultiDual{N, T}

A number type for chunked forward-mode automatic differentiation.
`N` is the "chunk size" - the number of derivatives computed simultaneously.
It holds a real part and a `dual` part, which is an `SVector` of length `N`.
"""
struct MultiDual{N, T} <: Number
    real::T
    dual::SVector{N, T}
end

# --- Constructors and Base functions ---
MultiDual{N, T}(x::Real) where {N, T} = MultiDual(T(x), zero(SVector{N, T}))
MultiDual(x::Real, dual_vec::SVector{N, T}) where {N, T} = MultiDual{N, T}(x, dual_vec)

# Needed for generic code
Base.real(d::MultiDual) = d.real
Base.zero(::Type{MultiDual{N, T}}) where {N, T} = MultiDual(zero(T), zero(SVector{N, T}))
Base.one(::Type{MultiDual{N, T}}) where {N, T} = MultiDual(one(T), zero(SVector{N, T}))
Base.conj(d::MultiDual) = d

# --- Promotion and Conversion ---
Base.promote_rule(::Type{MultiDual{N, T}}, ::Type{S}) where {N, T, S<:Real} = MultiDual{N, promote_type(T, S)}
function Base.convert(::Type{MultiDual{N, T}}, x::S) where {N, T, S<:Real}
    # This implementation correctly handles nested MultiDual types.
    # When T is itself a MultiDual, T(x) correctly "lifts" the real number x
    # into the nested type for the real part. We then provide a zero dual part.
    real_part = T(x)
    return MultiDual(real_part, zero(SVector{N, T}))
end
Base.convert(::Type{MultiDual{N, T}}, d::MultiDual{N, S}) where {N, T, S} = MultiDual(T(d.real), SVector{N, T}(d.dual))

# --- Core Arithmetic ---
Base.:+(a::MultiDual{N,T}, b::MultiDual{N,T}) where {N,T} = MultiDual(a.real + b.real, a.dual + b.dual)
Base.:+(a::MultiDual{N,T}, b::Real) where {N,T} = MultiDual(a.real + b, a.dual)
Base.:+(a::Real, b::MultiDual{N,T}) where {N,T} = b + a

Base.:-(a::MultiDual) = MultiDual(-a.real, -a.dual)
Base.:-(a::MultiDual, b::MultiDual) = a + (-b)

Base.:*(a::MultiDual{N,T}, b::MultiDual{N,T}) where {N,T} = MultiDual(a.real * b.real, a.real * b.dual + a.dual * b.real)
Base.:*(a::MultiDual{N,T}, b::Real) where {N,T} = MultiDual(a.real * b, a.dual * b)
Base.:*(a::Real, b::MultiDual{N,T}) where {N,T} = b * a

function Base.inv(a::MultiDual{N,T}) where {N,T}
    inv_a_real = inv(a.real)
    MultiDual(inv_a_real, -a.dual * (inv_a_real^2))
end

Base.:/(a::MultiDual, b::MultiDual) = a * inv(b)
Base.:/(a::MultiDual, b::Real) = MultiDual(a.real / b, a.dual / b)
Base.:/(a::Real, b::MultiDual) = a * inv(b)

# --- Exponentiation ---
Base.:^(a::MultiDual, n::Integer) = Base.power_by_squaring(a, n)
Base.:^(a::MultiDual, n::Real) = exp(n * log(a))

# --- Elementary Functions ---
function Base.sin(a::MultiDual{N,T}) where {N,T}
    s, c = sincos(a.real)
    MultiDual(s, c * a.dual)
end

function Base.cos(a::MultiDual{N,T}) where {N,T}
    s, c = sincos(a.real)
    MultiDual(c, -s * a.dual)
end

function Base.tan(a::MultiDual{N,T}) where {N,T}
    t = tan(a.real)
    MultiDual(t, a.dual * (1 + t^2))
end

function Base.tanh(a::MultiDual{N,T}) where {N,T}
    t = tanh(a.real)
    # Derivative of tanh(x) is sech(x)^2 = 1 - tanh(x)^2
    MultiDual(t, a.dual * (1 - t^2))
end

function Base.exp(a::MultiDual{N,T}) where {N,T}
    ex = exp(a.real)
    MultiDual(ex, ex * a.dual)
end

function Base.log(a::MultiDual{N,T}) where {N,T}
    MultiDual(log(a.real), a.dual / a.real)
end

function Base.sqrt(a::MultiDual{N,T}) where {N,T}
    s = sqrt(a.real)
    MultiDual(s, a.dual / (2s))
end

function Base.abs(a::MultiDual)
    # The derivative of abs(x) is sign(x).
    MultiDual(abs(a.real), sign(a.real) * a.dual)
end

"""
    acos(a::MultiDual)

Overloads the `acos` function for `MultiDual` numbers.
"""
function Base.acos(a::MultiDual{N,T}) where {N,T}
    x = real(a)
    # The derivative of acos(x) is -1/sqrt(1-x^2).
    # We assume the input `x` is already clamped to the valid domain [-1, 1]
    # by the calling function to avoid DomainErrors.
    deriv = -1 / sqrt(1 - x^2)
    MultiDual(acos(x), deriv * a.dual)
end

"""
    clamp(d::MultiDual, lo, hi)

Overloads the `clamp` function for `MultiDual` numbers.
"""
function Base.clamp(d::MultiDual{N, T}, lo::Real, hi::Real) where {N, T}
    r = real(d)
    if r < lo
        return MultiDual(T(lo), zero(SVector{N, T})) # Derivative is 0 outside the range
    elseif r > hi
        return MultiDual(T(hi), zero(SVector{N, T})) # Derivative is 0 outside the range
    end
    return d # Derivative is 1 inside the range
end

# --- High-Performance Gradient & Jacobian ---

"""
    gradient!(grad, f, X, ::Val{N}, workspace)

The core, non-allocating gradient computation function.
`N` is the chunk size, passed as a `Val` type for compile-time specialization.
"""
function gradient!(grad, f, X::AbstractVector{T}, ::Val{N}, workspace::AbstractVector{<:MultiDual}) where {T, N}
    len = length(X)
    I_N = SMatrix{N, N, T}(I)       # Seeding matrix
    zero_dual = zero(SVector{N, T}) # Pre-allocate a zero vector for dual parts

    # Seed the workspace with real values and zero duals ONCE at the beginning.
    @inbounds for k in 1:len; workspace[k] = MultiDual(X[k], zero_dual); end

    # Iterate over the input vector in chunks of size N
    for i in 1:N:len
        chunk_size = min(N, len - i + 1)

        # 1. Seed the current chunk with dual parts from the identity matrix
        @inbounds for j in 1:chunk_size
            workspace[i + j - 1] = MultiDual(X[i + j - 1], I_N[:, j])
        end

        # 2. Evaluate the function ONCE for this chunk
        result_dual = f(workspace)

        # 3. Extract the N derivatives from the result's dual part
        dual_parts = result_dual.dual
        @inbounds for j in 1:chunk_size
            grad[i + j - 1] = dual_parts[j]
        end

        # 4. Reset the dual parts of the chunk to zero for the next iteration.
        # This is cheaper than re-creating the entire workspace.
        @inbounds for j in 1:chunk_size
            workspace[i + j - 1] = MultiDual(X[i + j - 1], zero_dual)
        end
    end
    return grad
end

"""
    loss_and_gradient!(grad, f, X, ::Val{N}, workspace)

The core, non-allocating function to compute both the loss (scalar function value)
and the gradient simultaneously. This is more efficient than separate calls.
"""
function loss_and_gradient!(grad, f, X::AbstractVector{T}, ::Val{N}, workspace::AbstractVector{<:MultiDual}) where {T, N}
    len = length(X)
    I_N = SMatrix{N, N, T}(I)
    zero_dual = zero(SVector{N, T})
    loss = zero(T) # Initialize loss

    # Seed the workspace with real values and zero duals ONCE at the beginning.
    @inbounds for k in 1:len; workspace[k] = MultiDual(X[k], zero_dual); end

    # Iterate over the input vector in chunks of size N
    for i in 1:N:len
        chunk_size = min(N, len - i + 1)

        # 1. Seed the current chunk with dual parts from the identity matrix
        @inbounds for j in 1:chunk_size
            workspace[i + j - 1] = MultiDual(X[i + j - 1], I_N[:, j])
        end

        # 2. Evaluate the function ONCE for this chunk
        result_dual = f(workspace)
        if i == 1; loss = real(result_dual); end # The real part is the loss, capture it once.

        # 3. Extract derivatives and reset the chunk
        dual_parts = result_dual.dual
        @inbounds for j in 1:chunk_size
            grad[i + j - 1] = dual_parts[j]
            workspace[i + j - 1] = MultiDual(X[i + j - 1], zero_dual)
        end
    end
    return loss
end

"""
    jacobian!(J, f, X, ::Val{N}, workspace)

The core, non-allocating jacobian computation function.
`J` is the pre-allocated Jacobian matrix.
"""
function jacobian!(J, f, X::AbstractVector{T}, ::Val{N}, workspace::AbstractVector{<:MultiDual}) where {T, N}
    len_in = length(X)

    I_N = SMatrix{N, N, T}(I)
    zero_dual = zero(SVector{N, T})

    # Seed the workspace with real values and zero duals ONCE at the beginning.
    @inbounds for k in 1:len_in; workspace[k] = MultiDual(X[k], zero_dual); end

    # Iterate over the input vector in chunks of size N
    for i in 1:N:len_in
        chunk_size = min(N, len_in - i + 1)

        # 1. Seed the current chunk with dual parts from the identity matrix
        @inbounds for j in 1:chunk_size
            workspace[i + j - 1] = MultiDual(X[i + j - 1], I_N[:, j])
        end

        # 2. Evaluate the vector-valued function ONCE for this chunk
        result_dual_vector = f(workspace)

        # 3. Extract the dual parts into the corresponding columns of the Jacobian
        for k in eachindex(result_dual_vector) # Use eachindex for robustness
            dual_parts = result_dual_vector[k].dual
            for j in 1:chunk_size # Iterate over input dimensions in the chunk
                J[k, i + j - 1] = dual_parts[j]
            end
        end

        # 4. Reset the dual parts of the chunk to zero for the next iteration.
        @inbounds for j in 1:chunk_size
            workspace[i + j - 1] = MultiDual(X[i + j - 1], zero_dual)
        end
    end
    return J
end


# --- Configuration Objects for Non-Allocating API ---

"""
    GradientConfig(X; chunk_size=8)

A configuration object that holds pre-allocated workspace memory for `gradient!`.
Reusing this object in a loop avoids allocations.
"""
struct GradientConfig{N, T}
    workspace::Vector{MultiDual{N, T}}
end

function GradientConfig(X::AbstractVector{T}; chunk_size=8) where T
    N = chunk_size
    workspace = Vector{MultiDual{N, T}}(undef, length(X))
    return GradientConfig{N, T}(workspace)
end

"""
    JacobianConfig(f, X; chunk_size=8)

A configuration object that holds pre-allocated workspace memory for `jacobian!`.
"""
struct JacobianConfig{N, T, M}
    result::Matrix{T}
    workspace::Vector{MultiDual{N, T}}
end

function JacobianConfig(f, X::AbstractVector{T}; chunk_size=8) where T
    N = chunk_size
    Y_template = f(X)
    M = length(Y_template)
    result = similar(X, M, length(X))
    workspace = Vector{MultiDual{N, T}}(undef, length(X))
    return JacobianConfig{N, T, M}(result, workspace)
end

"""
    JacobianConfig(M, X; chunk_size=8)

A configuration object that holds pre-allocated workspace memory for `jacobian!`.
This constructor is for cases where calling `f(X)` during setup is not feasible.
`M` is the output dimension of the function.
"""
function JacobianConfig(M::Integer, X::AbstractVector{T}; chunk_size=8) where T
    N = chunk_size
    workspace = Vector{MultiDual{N, T}}(undef, length(X))
    return JacobianConfig{N, T, M}(similar(X, M, length(X)), workspace)
end

# --- User-Facing API ---

"""
    gradient!(grad, f, X, cfg::GradientConfig)

High-performance, non-allocating gradient computation using a pre-configured workspace.
"""
gradient!(grad, f, X, cfg::GradientConfig{N}) where {N} = gradient!(grad, f, X, Val(N), cfg.workspace)

"""
    loss_and_gradient!(grad, f, X, cfg::GradientConfig)

High-performance, non-allocating function to compute loss and gradient simultaneously
using a pre-configured workspace. Returns the scalar loss.
"""
loss_and_gradient!(grad, f, X, cfg::GradientConfig{N}) where {N} = loss_and_gradient!(grad, f, X, Val(N), cfg.workspace)

"""
User-facing function to compute the gradient using the MultiDual engine.
This version allocates memory on each call. For performance, create a `GradientConfig`
and use `gradient!`.
"""
function gradient(f, X::AbstractVector{T}; chunk_size=8) where T
    grad = similar(X, float(T)) # Use float(T) to avoid integer truncation
    cfg = GradientConfig(X; chunk_size=chunk_size)
    gradient!(grad, f, X, cfg)
    return grad
end

"""
    jacobian!(J, f, X, cfg::JacobianConfig)

High-performance, non-allocating Jacobian computation using a pre-configured workspace.
"""
jacobian!(J, f, X, cfg::JacobianConfig{N}) where {N} = jacobian!(J, f, X, Val(N), cfg.workspace)

"""
    jacobian(f, X; chunk_size=8)

User-facing function to compute the Jacobian using the MultiDual engine.
This version allocates memory on each call. For performance, create a `JacobianConfig`
and use `jacobian!`.
"""
function jacobian(f, X::AbstractVector{T}; chunk_size=8) where T
    cfg = JacobianConfig(f, X; chunk_size=chunk_size)
    # The result matrix is already allocated inside the config object.
    jacobian!(cfg.result, f, X, cfg)
    return cfg.result
end

end # module ChunkedAD