@testset "Integration Tests" begin
    @testset "AD with Manifolds" begin
        M = Sphere(3)
        
        # Function to minimize
        f(p) = p[3]  # z-coordinate
        
        # Test that we can compute gradient on manifold
        x0 = [1/√2, 1/√2, 0.0]
        grad = ChunkedAD.gradient(f, x0)
        tangent_grad = project_tangent(M, x0, grad)
        
        @test abs(dot(tangent_grad, x0)) < 1e-10  # Should be orthogonal to manifold
    end
end