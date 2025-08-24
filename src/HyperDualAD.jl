# ======================================================================
# HyperDual Automatic Differentiation (Second-Order)
# Author: Aditya
# Version: 1.4.0
# Features:
#   - Exact first/second derivatives via HyperDual numbers (ε₁²=ε₂²=ε₁ε₂=0)
#   - Gradient and Hessian computation for scalar/vector functions
#   - Optimized for CPU with minimal allocations
#   - Implements standard, proven HyperDual algebra for robust results
# ======================================================================

module HyperDualAD

export HyperDual, second_derivative, hessian, hessian!, gradient_hessian, gradient_hessian!, eps1, eps2, eps1eps2

using LinearAlgebra
using ..ScalarAD: Dual, dual

# --------------------------
# HyperDual Number Type
# --------------------------
# A hyper-dual number is a dual number of dual numbers.
# h = a + bε₁ + cε₂ + dε₁ε₂  is represented as  (a + bε₁) + (c + dε₁)ε₂
# This leverages the existing Dual number implementation from ScalarAD.jl
# for all arithmetic and elementary function overloads.
const HyperDual{T} = Dual{Dual{T}}

# Base constructors
HyperDual(r::T, e1::T, e2::T, e1e2::T) where {T<:Real} = Dual(Dual(r, e1), Dual(e2, e1e2))
HyperDual(x::T) where {T<:Real} = Dual(Dual(x, zero(T)), Dual(zero(T), zero(T)))
HyperDual{T}(r, e1, e2, e1e2) where {T<:Real} = HyperDual(T(r), T(e1), T(e2), T(e1e2))

# Component accessors
Base.real(h::HyperDual) = real(h.data[1])
eps1(h::HyperDual) = dual(h.data[1])
eps2(h::HyperDual) = real(h.data[2])
eps1eps2(h::HyperDual) = dual(h.data[2])

# All arithmetic, promotion, and base functions are inherited from ScalarAD.Dual
function Base.isapprox(a::HyperDual, b::HyperDual; kwargs...)
    isapprox(real(a), real(b); kwargs...) &&
    isapprox(eps1(a), eps1(b); kwargs...) &&
    isapprox(eps2(a), eps2(b); kwargs...) &&
    isapprox(eps1eps2(a), eps1eps2(b); kwargs...)
end

Base.promote_rule(::Type{HyperDual{T}}, ::Type{HyperDual{S}}) where {T,S} = HyperDual{promote_type(T,S)}
Base.promote_rule(::Type{HyperDual{T}}, ::Type{S}) where {T<:Real, S<:Real} = HyperDual{promote_type(T, S)}
Base.convert(::Type{HyperDual{T}}, x::S) where {T<:Real, S<:Real} = HyperDual(T(x))
Base.convert(::Type{HyperDual{T}}, d::HyperDual{S}) where {T,S} = HyperDual{T}(T(real(d)), T(eps1(d)), T(eps2(d)), T(eps1eps2(d)))

# Arithmetic with real numbers
Base.:+(a::HyperDual, b::Real) = HyperDual(real(a) + b, eps1(a), eps2(a), eps1eps2(a))
Base.:+(a::Real, b::HyperDual) = b + a
Base.:-(a::HyperDual, b::Real) = HyperDual(real(a) - b, eps1(a), eps2(a), eps1eps2(a))
Base.:-(a::Real, b::HyperDual) = HyperDual(a - real(b), -eps1(b), -eps2(b), -eps1eps2(b))
Base.:*(a::HyperDual, b::Real) = HyperDual(real(a) * b, eps1(a) * b, eps2(a) * b, eps1eps2(a) * b)
Base.:*(a::Real, b::HyperDual) = b * a

function Base.:*(a::HyperDual, b::HyperDual)
    ra, ea1, ea2, ea12 = real(a), eps1(a), eps2(a), eps1eps2(a)
    rb, eb1, eb2, eb12 = real(b), eps1(b), eps2(b), eps1eps2(b)
    return HyperDual(ra*rb, ra*eb1 + ea1*rb, ra*eb2 + ea2*rb, ra*eb12 + ea1*eb2 + ea2*eb1 + ea12*rb)
end

Base.inv(h::HyperDual) = exp(-log(h))
Base.:/(a::HyperDual, b::HyperDual) = a * inv(b)

# --------------------------
# Elementary Functions
# --------------------------
function Base.exp(h::HyperDual)
    r, e1, e2, e12 = real(h), eps1(h), eps2(h), eps1eps2(h)
    expr = exp(r)
    HyperDual(expr, expr*e1, expr*e2, expr*(e12 + e1*e2))
end

function Base.sin(h::HyperDual)
    r, e1, e2, e12 = real(h), eps1(h), eps2(h), eps1eps2(h)
    sinr, cosr = sin(r), cos(r)
    HyperDual(sinr, cosr*e1, cosr*e2, cosr*e12 - sinr*e1*e2)
end

function Base.cos(h::HyperDual)
    r, e1, e2, e12 = real(h), eps1(h), eps2(h), eps1eps2(h)
    sinr, cosr = sin(r), cos(r)
    HyperDual(cosr, -sinr*e1, -sinr*e2, -sinr*e12 - cosr*e1*e2)
end

function Base.log(h::HyperDual)
    r, e1, e2, e12 = real(h), eps1(h), eps2(h), eps1eps2(h)
    logr = log(r)
    HyperDual(logr, e1/r, e2/r, e12/r - e1*e2/r^2)
end

# --------------------------
# Derivative Computations
# --------------------------
function second_derivative(f, x)
    eps1eps2(f(HyperDual(x, 1.0, 1.0, 0.0)))
end

function hessian(f, X::AbstractVector{T}) where T
    n = length(X)
    H = zeros(T, n, n)
    workspace = [HyperDual(x) for x in X]
    hessian!(H, f, X, workspace)
    return H
end

function hessian!(H, f, X::AbstractVector, workspace::AbstractVector{<:HyperDual})
    fill!(H, 0)
    @inbounds for i in eachindex(X); workspace[i] = HyperDual(X[i]); end

    @inbounds for i in axes(H, 1)
        original_val = workspace[i]
        workspace[i] = HyperDual(real(original_val), 1.0, 1.0, 0.0)
        H[i, i] = eps1eps2(f(workspace))
        workspace[i] = original_val
    end
    _compute_off_diagonals!(f, H, workspace)
    return H
end

function _compute_off_diagonals!(f, H, workspace::AbstractVector{<:HyperDual})
    inds = axes(H, 1)
    @inbounds for i in inds
        for j in (i + 1):last(inds)
            orig_i, orig_j = workspace[i], workspace[j]
            workspace[i] = HyperDual(real(orig_i), 1.0, 0.0, 0.0)
            workspace[j] = HyperDual(real(orig_j), 0.0, 1.0, 0.0)
            H[i, j] = H[j, i] = eps1eps2(f(workspace))
            workspace[i], workspace[j] = orig_i, orig_j
        end
    end
end

function gradient_hessian(f, X::AbstractVector{T}) where T
    grad = zeros(T, length(X))
    H = zeros(T, length(X), length(X))
    workspace = [HyperDual(x) for x in X]
    gradient_hessian!(grad, H, f, X, workspace)
    return grad, H
end

function gradient_hessian!(grad, H, f, X::AbstractVector, workspace::AbstractVector{<:HyperDual})
    fill!(grad, 0)
    fill!(H, 0)
    @inbounds for i in eachindex(X); workspace[i] = HyperDual(X[i]); end

    @inbounds for i in eachindex(X)
        original_val = workspace[i]
        workspace[i] = HyperDual(real(original_val), 1.0, 1.0, 0.0)
        result = f(workspace)
        grad[i] = eps1(result)
        H[i, i] = eps1eps2(result)
        workspace[i] = original_val
    end
    _compute_off_diagonals!(f, H, workspace)
    return grad, H
end

end # module HyperDualAD