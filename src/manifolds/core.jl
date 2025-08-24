abstract type Manifold end

# Base methods that should be implemented for each manifold
function project(M::AbstractManifold, p) 
    error("project not implemented for manifold of type $(typeof(M))")
end

function retract(M::AbstractManifold, p, v) 
    error("retract not implemented for manifold of type $(typeof(M))")
end

function project_tangent(M::AbstractManifold, p, v) 
    error("project_tangent not implemented for manifold of type $(typeof(M))")
end

function get_basis(M::AbstractManifold, p) 
    error("get_basis not implemented for manifold of type $(typeof(M))")
end

# Include specific manifold implementations
include("sphere.jl")
include("stiefel.jl")
include("euclidean.jl")
include("product_manifold.jl")
include("so3.jl")