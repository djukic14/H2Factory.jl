module TestToeplitzPermutation

using Test
using FFTW
using LinearAlgebra
using StaticArrays
using H2Factory, H2Trees

# THE LATTICE SYMMETRY ACTS EXACTLY ON A TOEPLITZ DIFFERENCE TENSOR, AND SURVIVES THE FFT.
#
# This is the fact that makes the translation reduction available to a Toeplitz coupling, and it was
# WRONGLY DENIED for a while: a measurement on MLFMA's fourteen-direction diagonal coupling showed
# its DFT does not permute, and that was over-generalised into "the reduction is diagonal-specific".
# Fourteen arbitrary directions are not a regular grid and have no reason to behave under a DFT; a
# Toeplitz coupling stores something else, the transform of a difference-domain kernel ON A REGULAR
# TENSOR GRID, and there the permutation goes straight through. So this file measures the real
# object rather than arguing from the wrong one.
#
# WHY IT SURVIVES: the DFT is separable, so an axis permutation of the tensor is an axis permutation
# of its transform; and the circulant embedding stores difference `d` at index `d mod L`, making
# `d -> -d` a CIRCULAR reversal, which the DFT maps to a circular reversal in frequency. A signed
# coordinate permutation is exactly {axis permutation} composed with {reversals}, so every one of the
# 48 elements survives.

# A radially symmetric kernel, hence invariant under the lattice group, `k(Sx) == k(x)`, which is
# the hypothesis the identity needs. Complex, so a dropped conjugation would show.
kern(x) = exp(1im * 1.3 * norm(x)) / (norm(x) + 0.5)

# The stored convention `toeplitzpermutation` assumes, and the one `toeplitzmul!` multiplies
# against: axis i holds differences 0, 1, …, n-1, -(n-1), …, -1.
wraporder(n) = vcat(0:(n - 1), (-(n - 1)):-1)

function tensorat(displacement, spacing, n)
    m = wraporder(n)
    return [kern(displacement + spacing * SVector(a, b, c)) for a in m, b in m, c in m]
end

@testset "Toeplitz difference tensor: the symmetry acts exactly, in both domains" begin
    n = 4
    spacing = 1.0
    displacement = SVector(5.0, 2.0, -3.0)
    toeplitzsize = H2Factory.standardtoeplitzsize((n, n, n))
    @test toeplitzsize == (2n - 1, 2n - 1, 2n - 1)

    canonical = tensorat(displacement, spacing, n)
    spectrum = fft(canonical)
    group = H2Trees.symmetrygroup(H2Trees.FullLatticeSymmetry(), Val(3))
    @test length(group) == 48

    worstdifference = 0.0
    worstfrequency = 0.0
    for symmetry in group
        permutation = H2Factory.toeplitzpermutation(toeplitzsize, symmetry)
        # A genuine bijection, checked on every element, a "permutation" with a repeat would give
        # plausible numbers on a symmetric tensor and nonsense on a general one.
        @test sort(Int.(permutation)) == collect(1:prod(toeplitzsize))

        # The image tensor is built INDEPENDENTLY at `S d`, not derived from `canonical`. That is
        # what makes this a test of the identity rather than of its own restatement.
        image = tensorat(
            SVector(H2Trees.applysymmetry(symmetry, Tuple(displacement))), spacing, n
        )
        worstdifference = max(
            worstdifference, maximum(abs, vec(image) - vec(canonical)[permutation])
        )
        worstfrequency = max(
            worstfrequency, maximum(abs, vec(fft(image)) - vec(spectrum)[permutation])
        )
    end

    # EXACT in the difference domain: both sides evaluate `k` at the same points, in a different
    # order, so this is bit-for-bit rather than merely close.
    @test worstdifference == 0.0
    # And to round-off through the transform. Magnitudes here are ~27, so this is ~2.7e-16 relative.
    @test worstfrequency < 1e-13
end

@testset "The inverse convention is load-bearing" begin
    # `toeplitzpermutation` returns the action of `S⁻¹`, because the stored tag maps CANONICAL to
    # ACTUAL. Feeding it `S⁻¹` instead (so it computes `S`) must break, and only a CHIRAL element
    # can show that, since an involution is its own inverse and would pass either way. Without this,
    # every assertion above would hold under a reversed convention.
    n = 4
    toeplitzsize = H2Factory.standardtoeplitzsize((n, n, n))
    displacement = SVector(5.0, 2.0, -3.0)
    chiral = H2Trees.LatticeSymmetry((2, 3, 1), (1, -1, 1))
    @test H2Trees.inversesymmetry(chiral) != chiral

    canonical = tensorat(displacement, 1.0, n)
    image = tensorat(SVector(H2Trees.applysymmetry(chiral, Tuple(displacement))), 1.0, n)

    right = H2Factory.toeplitzpermutation(toeplitzsize, chiral)
    wrong = H2Factory.toeplitzpermutation(toeplitzsize, H2Trees.inversesymmetry(chiral))
    @test maximum(abs, vec(image) - vec(canonical)[right]) == 0.0
    @test maximum(abs, vec(image) - vec(canonical)[wrong]) > 1e-3
end

@testset "A degenerate tensor admits only the elements fixing its long axes" begin
    # NOT a formality, and the reason the existing 1-D Toeplitz fixtures cannot exercise this at all:
    # their tensor is `(2r-1, 1, 1)` while the TREE is three-dimensional, so H2Trees canonicalizes
    # with the full 48-element group and can legitimately hand back an axis-swapping element. Such an
    # element cannot act on a degenerate tensor. It would map a length-(2r-1) axis onto a length-1
    # one, so it must be refused rather than silently producing a permutation of the wrong grid.
    L = 7
    axisswap = H2Trees.LatticeSymmetry((2, 3, 1), (1, -1, 1))
    @test_throws ErrorException H2Factory.toeplitzpermutation((L, 1, 1), axisswap)
    @test_throws "cannot change an axis length" H2Factory.toeplitzpermutation(
        (L, 1, 1), axisswap
    )

    # An element that fixes the long axis is fine, and is a genuine permutation of it.
    reversal = H2Trees.LatticeSymmetry((1, 2, 3), (-1, 1, 1))
    permutation = H2Factory.toeplitzpermutation((L, 1, 1), reversal)
    @test sort(Int.(permutation)) == collect(1:L)
    # ... and it really is the reversal `d -> -d` in wrap-around order, not the identity.
    @test permutation != Int32.(1:L)
end

@testset "latticegridpermutation acts on a PLAIN grid, not a circulant one" begin
    # THE TWO PERMUTATIONS ARE NOT INTERCHANGEABLE, and this is the assertion that says so. A
    # Toeplitz coupling stores differences under a circulant embedding, where a sign flip is a
    # CIRCULAR reversal; a dense coupling stores values at grid POINTS, where it is a STRAIGHT one
    # (`i -> b+1-i`). Both are permutations of the right length, so no length or type check can tell
    # them apart, only this does.
    b = 5
    reflect = H2Trees.LatticeSymmetry((1, 2, 3), (1, 1, -1))
    grid = H2Factory.latticegridpermutation((b, b, b), reflect)
    circulant = H2Factory.toeplitzpermutation((b, b, b), reflect)
    @test length(grid) == length(circulant) == b^3
    @test sort(Int.(grid)) == collect(1:(b ^ 3))       # a genuine permutation
    @test grid != circulant                          # ... and NOT the circulant one

    # The straight reversal, stated directly on the axis it flips.
    linear = LinearIndices((b, b, b))
    for i in 1:b, j in 1:b, k in 1:b
        @test grid[linear[i, j, k]] == linear[i, j, b + 1 - k]
    end

    # A pure axis swap contains no reversal, so there the two DO agree, which is why a test built
    # only on rotations would miss the distinction entirely.
    swap = H2Trees.LatticeSymmetry((2, 1, 3), (1, 1, 1))
    @test H2Factory.latticegridpermutation((b, b, b), swap) ==
        H2Factory.toeplitzpermutation((b, b, b), swap)

    identity = H2Trees.LatticeSymmetry((1, 2, 3), (1, 1, 1))
    @test H2Factory.latticegridpermutation((b, b, b), identity) == Int32.(1:(b ^ 3))

    # An axis the element MIXES must not have a different length, or the element does not act on the
    # grid at all and a silently wrong permutation would come back.
    @test_throws ArgumentError H2Factory.latticegridpermutation(
        (b, b, b + 2), H2Trees.LatticeSymmetry((1, 3, 2), (1, 1, 1))
    )
end

end # module TestToeplitzPermutation
