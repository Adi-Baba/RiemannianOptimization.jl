using ..ScalarAD
using ..HyperDualAD

@testset "Autodiff Tests" begin
    @testset "Scalar AD" begin
        f(x) = x^2
        @test ScalarAD.derivative(f, 2.0) ≈ 4.0
        
        g(X) = X[1]^2 + X[2]^3
        X = [1.0, 2.0]
        @test ScalarAD.gradient(g, X) ≈ [2.0, 12.0]
    end
    
    @testset "HyperDual AD" begin
        f(x) = x^3
        @test HyperDualAD.second_derivative(f, 2.0) ≈ 12.0
        
        g(X) = X[1]^2 + X[2]^3
        X = [1.0, 2.0]
        grad, H = HyperDualAD.gradient_hessian(g, X)
        @test grad ≈ [2.0, 12.0]
        @test H ≈ [2.0 0.0; 0.0 12.0]
    end
    
    @testset "Chunked AD" begin
        g(X) = X[1]^2 + X[2]^3
        X = [1.0, 2.0]
        
        # Test gradient
        grad = ChunkedAD.gradient(g, X)
        @test grad ≈ [2.0, 12.0]
        
        # Test Jacobian
        h(X) = [X[1]^2, X[2]^3]
        J = ChunkedAD.jacobian(h, X)
        @test J ≈ [2.0 0.0; 0.0 12.0]
    end
end