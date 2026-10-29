module TestToeplitzI2O

using Test
using Random
using LinearAlgebra
using StaticArrays
using H2Factory
using H2Trees
using H2FactoryTestUtils

# Toeplitz far-field tests: the Toeplitz-moment analogue of the dense harness. The Toeplitz-specific
# machinery is the I2O TRANSLATION step (FFT modulation + frequency-domain multiply); the aggregate/
# disaggregate traversal is identical to the dense case (already covered). So we test the translation
# step directly, for every far pair, in all modes, via `H2FactoryTestUtils.checki2otranslation` --
# driving H2Factory's real pipeline `modulate_in! -> applyi2o! (Toeplitz coupling) -> modulate_out!` and
# comparing to the dense Toeplitz coupling. Galerkin (uniform AND non-uniform trees) and Petrov
# (`BlockTree`, three test/trial geometries), complex + real kernels, ranks 2/3/4.
#
# This is also the `couplingkey` REGRESSION GATE for the TOEPLITZ family (dense/diagonal are covered by
# test_i2oprotocol.jl): the real `ToeplitzI2OTranslationCollection` o2i/adjoint kernels now resolve their
# block through `couplingkey`, and a Toeplitz coupling is position-difference-based, so
# `S(recv,transl) != S(transl,recv)` -- a missing or double swap fails against the dense reference here,
# and the complex kernel makes transpose (`toeplitzmul!`) vs adjoint (`adjointtoeplitzmul!`, conj)
# distinguishable.
#
# Seeded for reproducibility.
Random.seed!(0x704e_1177)

mkt(v, mhs; mv=0) = H2Trees.buildtree(
    [SVector(c, 0.0, 0.0) for c in v];
    builder=H2Trees.TwoNTreeBuilder(; minhalfsize=mhs, root=1, minvalues=mv),
)
# Petrov `BlockTree` from separate test/trial 1D point sets (same coupling machinery, but the
# receiving node lives in the test tree and the translating node in the trial tree).
mkbt(tv, rv, mhs; tmv=0, rmv=0) = H2Trees.buildtree(
    [SVector(c, 0.0, 0.0) for c in tv],
    [SVector(c, 0.0, 0.0) for c in rv];
    builder=H2Trees.BlockTreeBuilder(;
        test=H2Trees.TwoNTreeBuilder(; minhalfsize=mhs, minvalues=tmv),
        trial=H2Trees.TwoNTreeBuilder(; minhalfsize=mhs, minvalues=rmv),
    ),
)
uniformpts(N) = [(i + 0.5) / N for i in 0:(N - 1)]
# Unbalanced: dense cluster (subdivides deep) + isolated points (shallow) -> leaves at many levels.
unbalancedpts() = vcat([0.002 * i for i in 0:31], [0.3, 0.5, 0.7, 0.9])

function nodelevel(tree, node)
    l = 1
    n = node
    while n != H2Trees.root(tree)
        n = H2Trees.parent(tree, n)
        l += 1
    end
    return l
end
leaflevels(tree) = sort(unique(nodelevel(tree, lf) for lf in H2Trees.leaves(tree)))

# Translation-invariant kernels k(x-y): complex-valued (adjoint != transpose is meaningful) and real.
complexkernel(z) = exp(im * z) / (1 + z^2)
realkernel(z) = 1 / (1 + z^2)

@testset "Toeplitz modulation policy" begin
    setup = toeplitzi2osetup(mkt(uniformpts(16), 0.05), complexkernel, 2)
    collection = setup.collection

    @test H2Factory.modulation(collection) isa H2Factory.ToeplitzModulation
    @test H2Factory.modulation(H2Factory.I2OTranslation(collection)) isa
        H2Factory.ToeplitzModulation
    @test_throws ArgumentError H2Factory.I2OTranslation(
        collection, H2Factory.NoModulation()
    )

    moment = zeros(ComplexF64, 2)
    args = (moment, 1, H2Factory.isForward(), H2Factory.NoModulation(), collection)
    @test_throws ArgumentError H2Factory.modulate_in!(args..., nothing, nothing, nothing)
    @test_throws ArgumentError H2Factory.modulate_out!(args..., nothing, nothing, nothing)
end

@testset "Toeplitz I2O translation vs dense coupling (all modes, complex + real kernels)" begin
    # Every far pair's FFT Toeplitz translation must reproduce its dense Toeplitz coupling S_ts, in
    # forward/transpose/adjoint mode. `checktoeplitzi2o` drives H2Factory's real pipeline via
    # `checki2otranslation` and @tests each (pair, mode).
    for (kname, kernel) in (("complex", complexkernel), ("real Float64", realkernel))
        @testset "$kname kernel" begin
            for rank in (2, 3, 4)
                @testset "rank $rank" begin
                    @test checktoeplitzi2o(mkt(uniformpts(64), 0.02), kernel, rank; n=3) > 0
                    unb = mkt(unbalancedpts(), 0.001; mv=1)
                    @test length(leaflevels(unb)) > 1      # genuinely non-uniform depth
                    @test checktoeplitzi2o(unb, kernel, rank; n=3) > 0
                end
            end
        end
    end
end

@testset "Petrov Toeplitz I2O translation vs dense coupling (all modes, complex + real kernels)" begin
    # The same FFT Toeplitz translation, but over a Petrov `BlockTree`: receiving nodes in the test
    # tree, translating nodes in the trial tree. `toeplitzi2osetup`/`checktoeplitzi2o` dispatch on the
    # tree type, using a same-BOX-SIZE (not same-level) far-pair check since a `BlockTree`'s two sides
    # can sit at offset levels. Cover the three user-specified test/trial geometries:
    #   coincident (test region == trial region), disjoint-far (separated clusters), overlapping.
    u = uniformpts(48)
    geometries = (
        ("coincident", u, u), ("disjoint-far", u, 100.0 .+ u), ("overlapping", u, 0.5 .+ u)
    )
    for (kname, kernel) in (("complex", complexkernel), ("real Float64", realkernel))
        @testset "$kname kernel" begin
            for (gname, tv, rv) in geometries
                @testset "$gname" begin
                    for rank in (2, 3)
                        @test checktoeplitzi2o(mkbt(tv, rv, 0.02), kernel, rank; n=3) > 0
                    end
                end
            end
        end
    end
end

@testset "Toeplitz structure of the couplings (doc §19.10)" begin
    # For a translation-invariant kernel + same-level boxes the couplings are Toeplitz and shared by
    # displacement. Verify on a uniform tree.
    tree = mkt(uniformpts(64), 0.02)
    rank = 3
    setup = toeplitzi2osetup(tree, complexkernel, rank)
    center1d(n) = H2Trees.center(tree, n)[1]
    # (a) each coupling block is Toeplitz: S[μ,ν] == S[μ+1,ν+1].
    for S in values(setup.couplings)
        @test all(S[μ, ν] ≈ S[μ + 1, ν + 1] for μ in 1:(rank - 1), ν in 1:(rank - 1))
    end
    # (b) same-level pairs with equal center displacement share the same coupling.
    bydisp = Dict{Tuple{Int,Float64},Matrix{ComplexF64}}()
    for (recv, transl) in setup.farpairs
        lvl = setup.levels[(recv, transl)]
        disp = round(center1d(recv) - center1d(transl); digits=10)
        S = setup.couplings[(recv, transl)]
        if haskey(bydisp, (lvl, disp))
            @test bydisp[(lvl, disp)] ≈ S      # Toeplitz reuse across blocks
        else
            bydisp[(lvl, disp)] = S
        end
    end
    @test !isempty(bydisp)
end

end # module TestToeplitzI2O
