# Helper functions for generic operations
_opnorm(x::AbstractVector) = norm(x)
_opnorm(x::AbstractMatrix) = norm(x)
_opdot(a::AbstractVector, b::AbstractVector) = dot(a, b)
_opdot(a::AbstractMatrix, b::AbstractMatrix) = dot(vec(a), vec(b))