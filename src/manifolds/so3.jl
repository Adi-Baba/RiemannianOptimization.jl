import Base: size

struct SO3 <: AbstractManifold end

function project(::SO3, X::AbstractMatrix)
    U, S, V = svd(X)
    return U * V'
end

function retract(::SO3, X::AbstractMatrix, V::AbstractMatrix)
    return project(SO3, X + V)
end

function project_tangent(::SO3, X::AbstractMatrix, V::AbstractMatrix)
    skew = V - V'
    return X * skew / 2
end

function inner(::SO3, X, U, V)
    return dot(U, V)
end

function distance(::SO3, p, q)
    error("distance not implemented for SO3 manifold")
end

function get_basis(::SO3, X::AbstractMatrix)
    # Basis for the tangent space of SO(3) at X
    G1 = [0 0 0; 0 0 -1; 0 1 0]
    G2 = [0 0 1; 0 0 0; -1 0 0]
    G3 = [0 -1 0; 1 0 0; 0 0 0]
    return [vec(X * G1) vec(X * G2) vec(X * G3)]
end

size(::SO3) = (3, 3)