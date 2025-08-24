# Export solver functions
export riemannian_gradient_descent, riemannian_trust_region

# Include solver implementations
include("gradient_descent.jl")
include("trust_region.jl")
include("utilities.jl")