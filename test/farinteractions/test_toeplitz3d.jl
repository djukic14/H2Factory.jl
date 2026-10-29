module TestToeplitz3D

using Test
using LinearAlgebra
using StaticArrays
using H2Factory, H2Trees
using H2FactoryTestUtils

# A NON-DEGENERATE TOEPLITZ COUPLING, AND WHY IT HAD TO BE BUILT.
#
# Every Toeplitz fixture before this one is one-dimensional: the difference tensor is `(2r-1, 1, 1)`.
# That exercises the FFT path, the modulation and the traversal perfectly well, because none of them
# care how many axes are non-degenerate. It cannot exercise the SYMMETRY reduction:
#
#   * the only lattice element such a tensor can represent is `d -> -d`, which is its own inverse --
#     so a symmetry test built on it would pass under a REVERSED inverse convention; and
#   * the tree is three-dimensional regardless, so canonicalization works with the full 48-element
#     group and can hand back an axis-swapping element the tensor cannot represent at all.
#
# WHAT THIS TESTSET ESTABLISHED, and it was worth measuring rather than assuming: the gap was
# entirely in the FIXTURE. H2Factory's Toeplitz machinery is already three-dimensional --
# `_copytobuffer!`/`_copytomoment!` are `Array{T,3}` views over whatever sizes they get, and `axes`
# takes `prod` over the per-axis node counts -- so a genuine `(2n-1)³` coupling goes through the REAL
# `ToeplitzI2OTranslationCollection` constructor and the real single-pair I2O with no change at all.

kern(z) = exp(1im * 1.1 * norm(z)) / (norm(z) + 0.7)

function cubetree(m=6; minhalfsize=1.0)
    points = [
        SVector(Float64(i), Float64(j), Float64(k)) for i in 0:(m - 1) for j in 0:(m - 1)
        for k in 0:(m - 1)
    ]
    return H2Trees.buildtree(
        points;
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=minhalfsize, minvalues=0, root=1),
    )
end

@testset "The tensor is genuinely three-dimensional" begin
    # NON-VACUITY FIRST, and here it is the whole point of the file: if this fixture were degenerate
    # in the same way the older ones are, everything below would still pass and prove nothing about
    # the case it was built for.
    tree = cubetree()
    nodesperaxis = 3
    setup = toeplitz3dsetup(tree, kern, nodesperaxis; nchunks=1)

    @test setup.rank == nodesperaxis^3
    @test !isempty(setup.farpairs)

    stored = setup.collection[first(setup.farpairs)][1].toeplitz
    @test size(stored) == (2nodesperaxis - 1, 2nodesperaxis - 1, 2nodesperaxis - 1)
    @test all(>(1), size(stored))                      # no singleton axis -- unlike the 1-D fixtures

    # And the coupling is a genuine multilevel Toeplitz matrix, not accidentally diagonal or
    # symmetric: a wrong permutation has to be able to show up as a wrong number.
    S = setup.couplings[first(setup.farpairs)]
    @test size(S) == (setup.rank, setup.rank)
    @test !isdiag(S)
    @test norm(S - transpose(S)) / norm(S) > 1e-6      # displacement is off-axis, so not symmetric
end

@testset "Real FFT Toeplitz path, all three modes, against the dense coupling" begin
    # The same check the one-dimensional fixture runs, on a coupling with three live axes. That it
    # passes unmodified is the evidence that H2Factory's Toeplitz path was never the 1-D part.
    tree = cubetree()
    @test checktoeplitz3di2o(tree, kern, 3; n=2, nchunks=1) > 0
end

@testset "A degenerate tensor cannot express a chiral permutation -- this one can" begin
    # THE BRIDGE TO THE SYMMETRY REDUCTION, and the quantitative reason the older fixtures cannot
    # substitute for this one. What matters is not how many group elements are ACCEPTED but how many
    # DISTINCT permutations they induce on the stored data, and how many of those are self-inverse:
    # a self-inverse permutation is exactly the case where using `S` where `S⁻¹` was meant is
    # invisible.
    #
    # MEASURED (and the counts are asserted, not described, so a change to the convention or the
    # embedding has to come here and say so):
    #
    #     (2n-1, 1, 1)   16 elements accepted, but only  2 distinct permutations, ALL self-inverse
    #     (2n-1)³        48 elements accepted,          48 distinct permutations, 20 self-inverse
    #
    # On a degenerate tensor the two short axes have length 1, so permuting them changes no index --
    # every admissible element collapses onto the identity or the reversal. A reversed-convention bug
    # is therefore PROVABLY undetectable there, not merely likely to slip through. The full tensor
    # leaves 28 chiral permutations, which is what makes a negative control possible at all.
    nodesperaxis = 3
    L = 2nodesperaxis - 1
    group = H2Trees.symmetrygroup(H2Trees.FullLatticeSymmetry(), Val(3))

    function inducedpermutations(toeplitzsize)
        accepted = filter(collect(group)) do symmetry
            try
                H2Factory.toeplitzpermutation(toeplitzsize, symmetry)
                return true
            catch
                return false
            end
        end
        distinct = unique([
            Int.(H2Factory.toeplitzpermutation(toeplitzsize, s)) for s in accepted
        ])
        return accepted, distinct
    end

    selfinverse(permutations) = count(p -> p[p] == collect(eachindex(p)), permutations)

    full, fulldistinct = inducedpermutations((L, L, L))
    @test length(full) == 48
    @test length(fulldistinct) == 48
    @test selfinverse(fulldistinct) == 20
    @test length(fulldistinct) - selfinverse(fulldistinct) == 28   # chiral, so a control is possible

    degenerate, degeneratedistinct = inducedpermutations((L, 1, 1))
    @test length(degenerate) == 16
    @test length(degeneratedistinct) == 2
    @test selfinverse(degeneratedistinct) == 2                     # ALL of them -- the whole problem
end

@testset "Symmetry-reduced Toeplitz: one tensor per orbit, applied through a permutation" begin
    # THE CLAIM THIS FILE EXISTS TO SETTLE. `#142` recorded that the translation reduction was
    # specific to `DiagonalTranslation`; that was wrong, and the evidence for it measured MLFMA's
    # fourteen-direction angular coupling rather than anything Toeplitz stores. Here the reduction
    # runs on a real Toeplitz collection, end to end.
    tree = cubetree()
    nodesperaxis = 3
    setup = symmetrytoeplitz3dsetup(tree, kern, nodesperaxis; nchunks=1)

    # NON-VACUITY. 19.75x is the full cubic group's ceiling over a level's far-field offsets, and
    # anything at or near 1 would mean the orbits never merged and nothing below is being tested.
    @test setup.stats.npairs == setup.plainstats.npairs
    @test setup.stats.nunique < setup.plainstats.nunique
    @test setup.plainstats.nunique / setup.stats.nunique > 5

    # THE STORED TENSORS MATCH, per pair, against a collection built independently for every
    # direction. Not bit-for-bit: the reduced path reads `κ_can[S⁻¹m]` where the unreduced one
    # evaluated `k` at `S·d + h·m` directly, so the two differ by round-off through the FFT.
    worst = maximum(setup.farpairs) do key
        reduced = collect(setup.reduced[key][1].toeplitz)
        plain = setup.plain[key][1].toeplitz
        return maximum(abs, reduced - plain) / maximum(abs, plain)
    end
    @test worst < 1e-12

    # THE TAGS ARE CONSUMED, so the reduced collection may be applied and the canonical one may not.
    # `assertnosymmetry` sees through the `ToeplitzI2OTranslationCollection` wrapper in both cases.
    @test H2Factory.assertnosymmetry(setup.reduced) === nothing
    @test_throws "symmetry-reduced" H2Factory.assertnosymmetry(setup.canonical)

    # AND IT APPLIES CORRECTLY, through the real FFT path, in all three directions.
    T = ComplexF64
    rank = setup.rank
    idmc = H2Factory.ToeplitzMomentCollection1D{T}((rank, rank))
    idmc[1] .= Matrix{T}(I, rank, rank)
    buildmom() = H2Factory.ToeplitzMoment1D{T}((nodesperaxis, nodesperaxis, nodesperaxis))
    xgen() = randn(T, rank)
    for key in setup.farpairs[1:min(60, end)]
        recv, transl = key
        S = setup.couplings[key]
        for (mode, reference) in (
            (H2Factory.isForward(), S),
            (H2Factory.isTranspose(), transpose(S)),
            (H2Factory.isAdjoint(), S'),
        )
            checki2otranslation(
                idmc,
                idmc,
                setup.reduced,
                setup.channel,
                recv,
                transl,
                setup.levels[key],
                buildmom,
                buildmom,
                xgen,
                Matrix(reference);
                mode=mode,
                n=2,
                ϵ=1e-9,
                verbose=false,
            )
        end
    end
end

@testset "The reversed convention is caught -- which the 1-D fixture could not do" begin
    # `toeplitzpermutation` returns the action of `S⁻¹`, because the tag maps CANONICAL to ACTUAL.
    # `useinverse=false` asks for `S` instead. On a `(2n-1)³` tensor 28 of the 48 induced
    # permutations are chiral, so the mistake shows; on the degenerate tensors the older fixtures
    # produce, every induced permutation is self-inverse and this control could not exist at all.
    tree = cubetree()
    wrong = symmetrytoeplitz3dsetup(tree, kern, 3; nchunks=1, useinverse=false)
    worst = maximum(wrong.farpairs) do key
        reduced = collect(wrong.reduced[key][1].toeplitz)
        plain = wrong.plain[key][1].toeplitz
        return maximum(abs, reduced - plain) / maximum(abs, plain)
    end
    @test worst > 1e-3
end

@testset "A builder must use the passed direction under the symmetry trait" begin
    # A CONTRACT WORTH PINNING, because violating it is silent. `computetranslations` calls the
    # builder once per translation ID with the ORBIT'S CANONICAL direction; the representative pair
    # whose node ids come with it generally has a DIFFERENT direction, related by its own tag. A
    # builder that ignores `direction` and recomputes from node centres therefore stores the
    # representative's tensor, and every consumer then permutes the wrong thing.
    #
    # Under `DirectionInvariance` the two agree and the mistake is invisible, which is exactly why
    # the one-dimensional fixture gets away with recomputing. Measured here: recomputing gives
    # stored tensors off by a relative 1.37 and every single-pair check fails.
    tree = cubetree()
    nodesperaxis = 3
    setup = symmetrytoeplitz3dsetup(tree, kern, nodesperaxis; nchunks=1)

    # The canonical direction handed to the builder is NOT the representative pair's own
    # displacement for at least some orbits -- if it always were, the contract would be vacuous.
    inner = setup.canonical.i2ocollection
    group = H2Factory.translationsymmetrygroup(inner)
    nonidentity = count(setup.farpairs) do key
        return H2Factory.translationsymmetry(inner, key[1], key[2]) !=
               H2Trees.identitysymmetry(Val(3))
    end
    @test nonidentity > 0
    @test group !== nothing
end

end # module TestToeplitz3D
