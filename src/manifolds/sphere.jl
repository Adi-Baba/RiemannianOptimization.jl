using LinearAlgebra
import Base: size

struct Sphere{N} <: AbstractManifold end

# Constructor for convenience
Sphere(n::Int) = Sphere{n}()

project(M::Sphere, p) = p / LinearAlgebra.norm(p)

retract(M::Sphere, p, v) = project(M, p + v)

function project_tangent(::Sphere, p, v)
    return v - dot(p, v) * p
end

function inner(::Sphere, p, u, v)
    return dot(u, v)
end

function distance(::Sphere, p, q)
    return acos(clamp(dot(p, q), -1.0, 1.0))
end

function get_basis(::Sphere, p::AbstractVector)
    # The tangent space is the nullspace of p'
    return nullspace(p')
end

size(::Sphere{N}) where N = (N + 1,)