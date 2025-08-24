# Manifolds

`RiemannianOptimization.jl` provides a set of built-in manifolds for common optimization problems. Each manifold is a subtype of the abstract `Manifold` type and implements a set of geometric operations.

## Supported Manifolds

### Euclidean

The `Euclidean(n)` manifold represents the n-dimensional Euclidean space. This is the standard setting for unconstrained optimization.

```julia
M = Euclidean(3)
```

### Sphere

The `Sphere(n)` manifold represents the (n-1)-dimensional sphere embedded in n-dimensional Euclidean space.

```julia
M = Sphere(3)  # Represents the 2-sphere in R^3
```

### Stiefel

The `Stiefel(n, p)` manifold represents the set of n-by-p orthonormal matrices. This is useful for problems with orthogonality constraints.

```julia
M = Stiefel(3, 2)  # Represents 3x2 orthonormal matrices
```

### SO(3)

The `SO(3)` manifold represents the Special Orthogonal group in 3 dimensions, which is the group of 3x3 rotation matrices.

```julia
M = SO3()
```

### ProductManifold

The `ProductManifold` allows you to combine multiple manifolds into a single product manifold.

```julia
M1 = Sphere(3)
M2 = Euclidean(2)
M = ProductManifold(M1, M2)
```

## Geometric Operations

Each manifold implements the following geometric operations:

- `project(M, p)`: Projects a point `p` onto the manifold `M`.
- `retract(M, p, v)`: Retracts a tangent vector `v` from a point `p` back to the manifold `M`.
- `project_tangent(M, p, v)`: Projects a vector `v` onto the tangent space of the manifold `M` at a point `p`.
- `inner(M, p, u, v)`: Computes the inner product of two tangent vectors `u` and `v` at a point `p`.
- `norm(M, p, v)`: Computes the norm of a tangent vector `v` at a point `p`.
- `distance(M, p, q)`: Computes the distance between two points `p` and `q` on the manifold `M`.
