using LinearAlgebra

@testset "Manifold Operations" begin
    @testset "Sphere Manifold" begin
        M = Sphere(3)
        p = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        
        # Test projection
        @test project(M, p) ≈ p
        @test LinearAlgebra.norm(project(M, [2.0, 0.0, 0.0])) ≈ 1.0
        
        # Test retraction
        @test retract(M, p, v) ≈ [1/√2, 1/√2, 0.0] atol=1e-6
        
        # Test tangent space projection
        @test project_tangent(M, p, v) ≈ v
        @test project_tangent(M, p, p) ≈ [0.0, 0.0, 0.0]
    end
    
    @testset "Euclidean Manifold" begin
        M = Euclidean(3)
        p = [1.0, 2.0, 3.0]
        v = [0.5, 0.5, 0.5]
        
        # All operations should be identity
        @test project(M, p) ≈ p
        @test retract(M, p, v) ≈ p + v
        @test project_tangent(M, p, v) ≈ v
    end
end