using RiemannianOptimization
using Test

println("Running RiemannianOptimization tests...")

@testset "Manifolds" begin
    include("test_manifolds.jl")
end

@testset "Solvers" begin
    include("test_solvers.jl")
end

@testset "Autodiff" begin
    include("test_autodiff.jl")
end

@testset "Integration" begin
    include("test_integration.jl")
end