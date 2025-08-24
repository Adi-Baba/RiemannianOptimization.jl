import Base: size

struct ProductManifold <: AbstractManifold
    manifolds::Vector{AbstractManifold}
    dimensions::Vector{Int}
    indices::Vector{UnitRange{Int}}
end

function ProductManifold(manifolds::AbstractManifold...)
    dims = [prod(size(m)) for m in manifolds]
    indices = Vector{UnitRange{Int}}(undef, length(manifolds))
    start_idx = 1
    
    for i in 1:length(manifolds)
        end_idx = start_idx + dims[i] - 1
        indices[i] = start_idx:end_idx
        start_idx = end_idx + 1
    end
    
    return ProductManifold(collect(manifolds), dims, indices)
end

function project(M::ProductManifold, p::AbstractVector)
    result = similar(p)
    for (i, m) in enumerate(M.manifolds)
        idx_range = M.indices[i]
        result[idx_range] .= vec(project(m, reshape(p[idx_range], size(m))))
    end
    return result
end

function retract(M::ProductManifold, p::AbstractVector, v::AbstractVector)
    result = similar(p)
    for (i, m) in enumerate(M.manifolds)
        idx_range = M.indices[i]
        p_part = reshape(p[idx_range], size(m))
        v_part = reshape(v[idx_range], size(m))
        result[idx_range] .= vec(retract(m, p_part, v_part))
    end
    return result
end

function project_tangent(M::ProductManifold, p::AbstractVector, v::AbstractVector)
    result = similar(v)
    for (i, m) in enumerate(M.manifolds)
        idx_range = M.indices[i]
        p_part = reshape(p[idx_range], size(m))
        v_part = reshape(v[idx_range], size(m))
        result[idx_range] .= vec(project_tangent(m, p_part, v_part))
    end
    return result
end

function inner(M::ProductManifold, p::AbstractVector, u::AbstractVector, v::AbstractVector)
    total_inner = 0.0
    for (i, m) in enumerate(M.manifolds)
        idx_range = M.indices[i]
        p_part = reshape(p[idx_range], size(m))
        u_part = reshape(u[idx_range], size(m))
        v_part = reshape(v[idx_range], size(m))
        total_inner += inner(m, p_part, u_part, v_part)
    end
    return total_inner
end

function distance(M::ProductManifold, p::AbstractVector, q::AbstractVector)
    total_distance_sq = 0.0
    for (i, m) in enumerate(M.manifolds)
        idx_range = M.indices[i]
        p_part = reshape(p[idx_range], size(m))
        q_part = reshape(q[idx_range], size(m))
        total_distance_sq += distance(m, p_part, q_part)^2
    end
    return sqrt(total_distance_sq)
end

function get_basis(M::ProductManifold, p::AbstractVector)
    total_dim = sum(M.dimensions)
    basis_blocks = []
    
    for (i, m) in enumerate(M.manifolds)
        idx_range = M.indices[i]
        p_part = reshape(p[idx_range], size(m))
        basis_block = get_basis(m, p_part)
        push!(basis_blocks, basis_block)
    end
    
    return BlockDiagonal(basis_blocks)
end

size(M::ProductManifold) = (sum(M.dimensions),)