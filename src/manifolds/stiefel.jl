import Base: size

struct Stiefel{N, P} <: AbstractManifold end

# Constructor for convenience
Stiefel(n::Int, p::Int) = Stiefel{n, p}()

function project(::Stiefel, X::AbstractMatrix)
    Q, R = qr(X)
    return Q * sign.(diag(R))
end

function retract(::Stiefel, X::AbstractMatrix, V::AbstractMatrix)
    Q, R = qr(X + V)
    return Q[:, 1:size(X, 2)]
end

function project_tangent(::Stiefel, X::AbstractMatrix, V::AbstractMatrix)
    return V - X * (X' * V + V' * X) / 2
end

function inner(::Stiefel, X, U, V)
    return dot(U, V)
end

function distance(::Stiefel, p, q)
    error("distance not implemented for Stiefel manifold")
end

function get_basis(::Stiefel, X::AbstractMatrix)
    # This is a simplified implementation
    # A proper implementation would generate a basis for the tangent space
    n, p = size(X)
    return Matrix(I, n, p)  # Simplified for demonstration
end

size(::Stiefel{N, P}) where {N, P} = (N, P)