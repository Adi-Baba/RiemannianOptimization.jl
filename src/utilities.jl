using LinearAlgebra

export solve_tr_subproblem

function solve_tr_subproblem(g::AbstractVector, H, Δ::Real)
    # Internal helper to find the positive root `τ` such that ||z + τd|| = Δ
    function _solve_to_boundary(z, d, Δ)
        a = dot(d, d)
        b = 2 * dot(z, d)
        c = dot(z, z) - Δ^2
        discriminant = b^2 - 4*a*c
        # Clamp to zero to prevent DomainError from tiny floating-point inaccuracies
        τ = (-b + sqrt(max(0, discriminant))) / (2a)
        return z + τ * d
    end

    T = eltype(g)
    z = zeros(T, length(g))
    r = g
    d = -r
    
    if LinearAlgebra.norm(g) < 1e-12
        return z
    end

    for _ in 1:length(g) # Max iterations is the dimension of the space
        Hd = H(d)
        d_Hd = dot(d, Hd)

        # If curvature is non-positive or zero, move along `d` to the boundary
        if d_Hd <= 0
            return _solve_to_boundary(z, d, Δ)
        end

        α = dot(r, r) / d_Hd
        z_new = z + α * d

        # If we step outside the trust region, move to the boundary and stop
        if LinearAlgebra.norm(z_new) >= Δ
            return _solve_to_boundary(z, d, Δ)
        end

        r_new = r + α * Hd
        if LinearAlgebra.norm(r_new) < 1e-12 * LinearAlgebra.norm(g)
            return z_new
        end

        β = dot(r_new, r_new) / dot(r, r)
        d = -r_new + β * d
        r = r_new
        z = z_new
    end
    return z
end