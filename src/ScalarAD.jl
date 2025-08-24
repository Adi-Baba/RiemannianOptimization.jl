# ======================================================================
# Module: ScalarAD
# Description: A simple, scalar (one-by-one) forward-mode automatic
#              differentiation implementation using dual numbers.
# ======================================================================
# Dual-Number Automatic Differentiation (CPU-Optimized)
# Author: Aditya
# Version: 1.0 (2025)
# Features:
#   - Exact derivative computation via dual numbers (ε²=0)
#   - Full operator/math function overloading
#   - Scalar/vector/tensor support
#   - CPU-parallelized gradient calculation
#
# Note: This module is designed to be loaded once per Julia session.
#       Repeated 'include' or 'using' without restarting the session
#       can lead to 'redefinition of constant' warnings/errors.
# ======================================================================

module ScalarAD

export derivative, gradient, jacobian, Dual

using StaticArrays
using LinearAlgebra

# Using an immutable struct with StaticArrays is significantly more performant.
# It allows for stack allocation and can enable SIMD optimizations.
struct Dual{T} <: Number
    data::SVector{2, T} # [real, dual]
end

# Accessors for clarity and easy transition
Base.real(d::Dual) = d.data[1]
dual(d::Dual) = d.data[2]

# Base constructors and essential numeric functions
Dual(real, dual) = Dual(SVector(real, dual))
Dual{T}(real, dual) where T = Dual(SVector{2, T}(real, dual))
Dual(x::Real) = Dual(x, zero(x))

# Promotion and conversion for better interoperability
Base.promote_rule(::Type{Dual{T}}, ::Type{S}) where {T<:Real, S<:Real} = Dual{promote_type(T, S)}
Base.convert(::Type{Dual{T}}, x::S) where {T, S<:Real} = Dual{T}(T(x), zero(T))
Base.convert(::Type{Dual{T}}, d::Dual{S}) where {T, S} = Dual(SVector{2, T}(d.data))

Base.zero(::Type{Dual{T}}) where T = Dual(zero(T), zero(T))
Base.zero(d::Dual) = zero(typeof(d))
Base.one(::Type{Dual{T}}) where T = Dual(one(T), zero(T))
Base.one(d::Dual) = one(typeof(d))

# --------------------------
# Core Operations
# --------------------------

# For compatibility with functions like dot()
Base.conj(d::Dual) = d

Base.:+(a::Dual, b::Dual) = Dual(a.data + b.data)
Base.:+(a::Dual, b::Number) = Dual(real(a) + b, dual(a))
Base.:+(a::Number, b::Dual) = Dual(a + real(b), dual(b))

Base.:-(a::Dual) = Dual(-a.data)
Base.:-(a::Dual, b::Dual) = a + (-b)

Base.:*(a::Dual, b::Dual) = Dual(real(a) * real(b), real(a)*dual(b) + dual(a)*real(b))
Base.:*(a::Dual, b::Number) = Dual(real(a) * b, dual(a) * b)
Base.:*(a::Number, b::Dual) = Dual(a * real(b), a * dual(b))

Base.:/(a::Dual, b::Dual) = Dual(real(a)/real(b), (dual(a)*real(b) - real(a)*dual(b))/(real(b)^2))
Base.:/(a::Dual, b::Real) = Dual(a.data / b)
Base.:/(a::Real, b::Dual) = Dual(a/real(b), -a*dual(b)/(real(b)^2))

Base.:^(a::Dual, n::Integer) = Base.power_by_squaring(a, n)
Base.:^(a::Dual, n::Real) = exp(n * log(a))

# --------------------------
# Elementary Functions
# --------------------------
Base.exp(a::Dual) = (ex = exp(real(a)); Dual(ex, ex * dual(a)))
Base.log(a::Dual) = Dual(log(real(a)), dual(a) / real(a))

Base.sin(a::Dual) = Dual(sin(real(a)), cos(real(a)) * dual(a))
Base.cos(a::Dual) = Dual(cos(real(a)), -sin(real(a)) * dual(a))
Base.tan(a::Dual) = (t = tan(real(a)); Dual(t, dual(a) * (1 + t^2)))
Base.acos(a::Dual) = Dual(acos(real(a)), -dual(a) / sqrt(1 - real(a)^2))
Base.tanh(a::Dual) = (t = tanh(real(a)); Dual(t, dual(a) * (1 - t^2)))

Base.sqrt(a::Dual) = (s = sqrt(real(a)); Dual(s, dual(a)/(2s)))
Base.abs(a::Dual) = Dual(abs(real(a)), sign(real(a)) * dual(a))

# Added for compatibility with clamp()
Base.clamp(d::Dual, lo::Real, hi::Real) = begin
    r = real(d)
    if r < lo
        return Dual(lo, zero(lo))
    elseif r > hi
        return Dual(hi, zero(hi))
    else
        return d
    end
end

# Define norm for arrays of Dual numbers to prevent type conversion errors
LinearAlgebra.norm(v::AbstractArray{<:Dual}) = sqrt(sum(abs2, v))

# --------------------------
# Derivative Computation
# --------------------------
"""
    derivative(f, x)
Compute exact derivative of f at x using dual numbers
"""
derivative(f, x) = dual(f(Dual(x, one(x))))

"""
    gradient(f, X)
Compute gradient vector of f at point X (vector input)
"""
function gradient(f, X::AbstractVector)
    n = length(X)
    grad = similar(X, float(eltype(X)), n) # Use float type to avoid integer truncation
    # The user-facing function allocates the workspace once.
    workspace = [Dual(x) for x in X]
    # It then calls the efficient, non-allocating version.
    gradient!(grad, f, X, workspace)
    return grad
end

"""
    gradient!(grad, f, X, workspace)

Mutating, non-allocating version of `gradient`. This is the high-performance core.
"""
function gradient!(grad, f, X::AbstractVector, workspace::AbstractVector{<:Dual})
    # Initialize workspace with real values and zeroed dual parts
    @inbounds for i in eachindex(X); workspace[i] = Dual(X[i]); end

    # Loop to compute each partial derivative. This is now allocation-free.
    @inbounds for i in eachindex(X)
        # Perturb the i-th element, compute, and restore.
        original_val = workspace[i]
        workspace[i] = Dual(real(original_val), one(X[i]))
        grad[i] = dual(f(workspace))
        workspace[i] = original_val
    end
end

# --------------------------
# Performance Extensions
# --------------------------
"""
    jacobian(f, X)
Compute Jacobian matrix for vector-valued function f
"""
function jacobian(f, X::AbstractVector)
    # Determine the output dimension (m) by evaluating the function once.
    Y_template = f(X)
    m = length(Y_template)
    n = length(X)
    # Use float type to avoid integer truncation in the Jacobian matrix
    J = similar(Y_template, float(eltype(X)), m, n)
    workspace = [Dual(x) for x in X]

    # This implementation computes the Jacobian column-by-column, which is
    # significantly more efficient for forward-mode AD as it only requires
    # n passes of the function `f`, where n is the input dimension.
    @inbounds for j in 1:n # Iterate through columns of the Jacobian
        @inbounds for k in 1:n; workspace[k] = Dual(X[k]); end # Reset workspace
        # Perturb the j-th input variable
        workspace[j] = Dual(X[j], 1.0)

        # A single evaluation of f gives the j-th column of the Jacobian
        Y_dual = f(workspace)

        # Extract the dual parts into the j-th column of J
        for i in 1:m; J[i, j] = dual(Y_dual[i]); end
    end
    return J
end

end # module ScalarAD