module RiemannianOptimization

using LinearAlgebra

# Core Abstract Interface
include("core.jl")

# Helper functions and types
include("utilities.jl")

# Manifold Implementations
include("manifolds/euclidean.jl")
include("manifolds/sphere.jl")
include("manifolds/stiefel.jl")
include("manifolds/product_manifold.jl")
include("manifolds/so3.jl")

# Solver Implementations
include("solvers/gradient_descent.jl")
include("solvers/trust_region.jl")

# Automatic Differentiation Implementations
include("ScalarAD.jl")
include("HyperDualAD.jl")
include("ChunkedAD.jl")

# Export the public API

# Manifolds
export AbstractManifold, Euclidean, Sphere, Stiefel, SO3, ProductManifold

# Geometric Operations
export retract, project, inner, norm, distance, project_tangent

# Solvers
export gradient_descent, riemannian_trust_region, ChunkedAD

# Automatic Differentiation
export ScalarAD, HyperDualAD

end