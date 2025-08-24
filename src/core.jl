export AbstractManifold, OptimizationResult

"""
    AbstractManifold

An abstract type for representing Riemannian manifolds.
"""
abstract type AbstractManifold end

"""
    OptimizationResult

A struct to hold the results of an optimization.
"""
struct OptimizationResult
    solution
    cost
    converged::Bool
    message::String
end

# Generic functions that should be implemented by each manifold
function project(M::AbstractManifold, p)
    error("project not implemented for manifold of type $(typeof(M)) and point of type $(typeof(p))")
end

function retract(M::AbstractManifold, p, v)
    error("retract not implemented for manifold of type $(typeof(M)), point of type $(typeof(p)), and vector of type $(typeof(v))")
end

function project_tangent(M::AbstractManifold, p, v)
    error("project_tangent not implemented for manifold of type $(typeof(M)), point of type $(typeof(p)), and vector of type $(typeof(v))")
end

function inner(M::AbstractManifold, p, u, v)
    error("inner not implemented for manifold of type $(typeof(M)), point of type $(typeof(p)), and vectors of type $(typeof(u)) and $(typeof(v))")
end

function norm(M::AbstractManifold, p, v)
    return sqrt(inner(M, p, v, v))
end

function distance(M::AbstractManifold, p, q)
    error("distance not implemented for manifold of type $(typeof(M)) and points of type $(typeof(p)) and $(typeof(q))")
end