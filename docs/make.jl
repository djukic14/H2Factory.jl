using H2Factory
using Documenter

DocMeta.setdocmeta!(H2Factory, :DocTestSetup, :(using H2Factory); recursive=true)

makedocs(;
    modules=[H2Factory],
    authors="Danijel Jukić <danijel.jukic14@gmail.com> and contributors",
    sitename="ℋ² Factory.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://djukic14.github.io/H2Factory.jl",
        edit_link="main",
        assets=["assets/diagrams.css"],
    ),
    # The reference spans several pages. Missing or duplicated docstrings must fail the build.
    checkdocs=:all,
    warnonly=false,
    pages=[
        "Home" => "index.md",
        "How the multiplication works" => "algorithm.md",
        "Construction and execution" => "construction.md",
        "Traits and dispatch" => "traits.md",
        "Maps" => [
            "Core and workspace" => "maps/core-workspace.md",
            "Accessors and orientation" => "maps/accessors.md",
            "Scheduling and multithreading" => "maps/scheduling.md",
        ],
        "Moments" => [
            "Moments and collections" => "moments/overview.md",
            "Creating and linking types" => "moments/types.md",
            "Allocation and storage reuse" => "moments/allocation.md",
            "Toeplitz moments" => "moments/toeplitz.md",
        ],
        "Near interactions" => [
            "Computing near interactions" => "near/assembly.md",
            "Storage and multiplication" => "near/storage.md",
        ],
        "Far interactions" => [
            "I2I and O2O level transfers" => "far/level-transfers.md",
            "I2O coupling" => "far/coupling.md",
            "Toeplitz coupling" => "far/toeplitz.md",
            "Symmetry reduction" => "far/symmetry.md",
            "Modulation and scratch" => "far/modulation.md",
            "Aggregation and disaggregation" => "far/traversal.md",
        ],
        "Implementing a fast method" => [
            "Required methods" => "implementing/required.md",
            "Worked example" => "implementing/example.md",
            "Optional capabilities" => "implementing/optional.md",
            "Testing an implementation" => "implementing/testing.md",
        ],
        "Diagnostics" => [
            "Structural checks" => "diagnostics/checks.md",
            "Structure and timing" => "diagnostics/reports.md",
        ],
        "Assembly extensions" => "extensions.md",
        "API Reference" => [
            "Overview" => "apiref.md",
            "Types and data containers" => "reference/types.md",
            "Functions" => "reference/functions.md",
            "Workspaces and scheduling" => "reference/execution.md",
            "Moment pools and aliasing" => "reference/pools.md",
            "Moment multiplication" => "reference/multiplication.md",
            "Conjugated and diagonal multiplication" => "reference/moment-operations.md",
            "Type-generation macros" => "reference/macros.md",
            "Extension declarations" => "reference/protocol.md",
            "Diagnostics" => "reference/diagnostics.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/djukic14/H2Factory.jl",
    target="build",
    devbranch="main",
    push_preview=true,
    forcepush=true,
    versions=["stable" => "v^", "dev" => "dev"],
)
