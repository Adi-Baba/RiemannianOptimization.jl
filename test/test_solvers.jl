include("../src/ChunkedAD.jl")
using LinearAlgebra

@testset "Solver Tests" begin
    @testset "Sphere Optimization" begin
        M = Sphere(3)
        
        # Minimize z-coordinate on sphere
        f(p) = p[3]
        x0 = [1/√2, 1/√2, 0.0]
        
        result = gradient_descent(M, f, p -> ChunkedAD.gradient(f, p), x0; step_size=0.005, max_iters=10000, grad_tol=1e-6)
        
        @test result.converged
        @test abs(result.solution[3] + 1.0) < 1e-4  # Should converge to south pole
    end
    
    @testset "Euclidean Optimization" begin
        M = Euclidean(2)
        
        # Simple quadratic function
        f(x) = x[1]^2 + x[2]^2
        x0 = [2.0, 2.0]
        
        result = gradient_descent(M, f, p -> ChunkedAD.gradient(f, p), x0; step_size=0.1, max_iters=100, grad_tol=1e-6)
        
        @test result.converged
        @test LinearAlgebra.norm(result.solution) < 1e-4
    end
end