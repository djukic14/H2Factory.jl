using Test
using H2Factory

# Tests use the `module Test*` + plain `@testset` convention (mirroring iFMM), not
# TestItems/TestItemRunner: every leaf test file is its own module (per-file isolation), each
# subdirectory has a `runtests.jl` grouping its leaves under a named `@testset`, and this top-level
# file includes those under one outer `@testset` plus the quality gates.
@testset verbose = true "H2Factory" begin
    include("maps/runtests.jl")
    include("moments/runtests.jl")
    include("H2BEAST/runtests.jl")
    include("H2Trees/runtests.jl")
    include("nearinteractions/runtests.jl")
    include("farinteractions/runtests.jl")
    include("i2otranslationcollection/runtests.jl")
    include("H2BlockSparse/runtests.jl")
    include("H2Toeplitz/runtests.jl")
    include("diagnostics/runtests.jl")
    include("Package/runtests.jl")

    @testset "Code quality (Aqua.jl)" begin
        using Aqua
        # `unbound_args=false`: the flagged methods are all `MOMENT{D,T}(size::NTuple{N,I})`-style
        # constructors (createmoment.jl/toeplitzmoment.jl) where `N`/`I` are only pinned down via
        # the `NTuple{N,I}` argument type -- Aqua's check trips on the classic empty-tuple edge case
        # (`NTuple{0,I}` doesn't actually constrain `I`), which is never exercised in practice
        # (moments are always constructed with a concrete nonzero-length size tuple). Same call as
        # iFMM's identically-shaped constructors, scoped out there for the same reason.
        Aqua.test_all(H2Factory; unbound_args=false)
    end

    @testset "Code formatting (JuliaFormatter.jl)" begin
        using JuliaFormatter
        @test JuliaFormatter.format(pkgdir(H2Factory), overwrite=false)
    end
end
