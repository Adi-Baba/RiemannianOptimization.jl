# Documentation build script for RiemannianOptimization.jl

using Documenter
using RiemannianOptimization

# Generate documentation
makedocs(
    sitename = "RiemannianOptimization.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Adi-Baba.github.io/RiemannianOptimization.jl",
        assets = ["assets/favicon.ico"],
    ),
    modules = [RiemannianOptimization],
    pages = [
        "Home" => "index.md",
        "Manifolds" => "manifolds.md",
        "Solvers" => "solvers.md",
        "Automatic Differentiation" => "autodiff.md",
        "Examples" => "examples.md",
    ]
)

# Deploy documentation to GitHub Pages
deploydocs(
    repo = "github.com/Adi-Baba/RiemannianOptimization.jl.git",
    devbranch = "main",
    push_preview = true
)