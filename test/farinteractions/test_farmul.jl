module TestFarmul

using Test
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils

@testset "Far interactions (synthetic, Galerkin, forward/non-Toeplitz)" begin
    # Two well-separated clusters of 2 points, each split into two sub-clusters. Chosen so the
    # resulting `TwoNTree` is a simple 5-level chain (root -> 1 child -> 1 child -> 2 children ->
    # leaves) whose only translating (far) pair sits at level 3 -- exercising two levels of I2I
    # aggregation (leaf->4->3), one level of I2O translation (3<->8), and two levels of O2O
    # disaggregation (3->4->leaf), all inside a tree tiny enough to hand-verify.
    pts = SVector{3,Float64}[]
    for x in (0.0, 0.1)
        push!(pts, SVector(x, 0.0, 0.0))
    end
    for x in (5.0, 5.1)
        push!(pts, SVector(x, 0.0, 0.0))
    end
    for x in (40.0, 40.1)
        push!(pts, SVector(x, 0.0, 0.0))
    end
    for x in (45.0, 45.1)
        push!(pts, SVector(x, 0.0, 0.0))
    end
    N = length(pts)

    tree = H2Trees.buildtree(
        pts; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=2.0, minvalues=0, root=1)
    )

    # Synthetic translators -- not real FMM physics (that lives in iFMM, out of scope here), just
    # enough structure to exercise H2Factory's tree-traversal/dispatch/accumulation logic with a
    # hand-computable result. I2I aggregates by plain summation (parent charge = sum of children's
    # charges, via `identitytranslator()`'s I2I role: copy unchanged, so accumulation across
    # children produces the sum), O2O disaggregation broadcasts a parent's accumulated far value
    # unchanged to its children (the same `identitytranslator()`, O2O reconstructed as
    # transpose/adjoint of I2I), and I2O multiplies by one fixed complex constant `c`
    # (`scalartranslator(c)`) -- chosen non-real so an accidentally dropped/misplaced conjugation
    # in the adjoint path would show up numerically instead of cancelling out.
    c = 2.0 - 1.0im
    T = ComplexF64

    A = fakegalerkinmap(
        tree;
        T=T,
        i2i=identitytranslator(),
        o2o=identitytranslator(),
        i2o=scalartranslator(c),
        scheduler=SerialScheduler(),
    )

    # Hand-derived dense far-field matrix for this setup: points 1-4 are the "left" cluster
    # (descendants of translating node 3), points 5-8 the "right" cluster (descendants of
    # translating node 8). I2I sums each cluster's charge unchanged up to the translating level,
    # I2O scales by `c`, and O2O broadcasts that scaled value unchanged back down to every point
    # in the *other* cluster -- so each point's far contribution is exactly `c` times the sum of
    # the opposite cluster's `x` values, with `α(A) == 1` folded in.
    F = zeros(T, N, N)
    F[1:4, 5:8] .= c
    F[5:8, 1:4] .= c

    x = randn(T, N)

    @testset "farmul! (non-accumulating)" begin
        y = Vector{T}(undef, N)
        @test H2Factory.farmul!(y, A, x) ≈ F * x
    end

    @testset "farmul! (transpose)" begin
        # `F` is symmetric by construction (the two clusters play interchangeable roles), so
        # `transpose(F) == F`; this checks that the transpose dispatch path reproduces it rather
        # than re-deriving it independently.
        y = Vector{T}(undef, N)
        @test H2Factory.farmul!(y, transpose(A), x) ≈ transpose(F) * x
    end

    @testset "farmul! (adjoint)" begin
        # `c` is non-real, so `adjoint(F) = conj(F) != F` -- this is only a meaningful check
        # because `c`'s imaginary part is nonzero.
        y = Vector{T}(undef, N)
        @test H2Factory.farmul!(y, adjoint(A), x) ≈ F' * x
    end

    @testset "farmul! (α, β accumulation)" begin
        # The 5-arg entry point is the one `LinearMaps._unsafe_mul!` actually calls during a full
        # `mul!`; it must *accumulate* into a pre-populated `y`, not overwrite it (unlike the
        # 3-arg entry point, which the "why" comment in `farmul.jl` explains zeroes `y` first).
        yaccumulate = randn(T, N)
        yref = copy(yaccumulate)
        α_, β_ = 0.7 + 0.3im, -0.4 + 0.2im
        @test H2Factory.farmul!(yaccumulate, A, x, α_, β_) ≈ α_ * F * x + β_ * yref
    end

    # `disaggregation.jl:407` (aggregation into the receiving node's moment after translation)
    # and `aggregation.jl:609` (aggregation-side I2O modulation) both end in `#TODO: iFFT ...` --
    # the FFT/Toeplitz aggregation-disaggregation path is genuinely incomplete production code,
    # not something this synthetic test can exercise meaningfully. Left untested here, same
    # "defer in-flight numerical logic rather than guess-fix" call as `test/runtests.jl`'s
    # `"Moment utils"` group makes for `toeplitzmul!`.
end

end # module TestFarmul
