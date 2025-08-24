import Base: size
using LinearAlgebra

"""
    Euclidean(n)

A struct representing the n-dimensional Euclidean space.
"""
struct Euclidean <: AbstractManifold
    n::Int
end

# The inner product is the standard dot product.
inner(::Euclidean, p, X, Y) = dot(X, Y)

# Projection onto the tangent space is the identity, as the tangent space is the space itself.
project(::Euclidean, p) = p

# A standard retraction on Euclidean space is vector addition.
retract(::Euclidean, p, X) = p + X

# The distance is the standard Euclidean norm of the difference.
distance(::Euclidean, p, q) = LinearAlgebra.norm(p - q)

project_tangent(::Euclidean, p, v) = v

size(M::Euclidean) = (M.n,)