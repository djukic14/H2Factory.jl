module TestReducedToeplitz

using Test
using LinearAlgebra
using H2Factory, H2Trees

# BUILDING THE REDUCED STORE FOR A TOEPLITZ COLLECTION.
#
# `test_componentaction.jl` checks the RULES (`componentsource`, `toeplitzpermutation`) against
# kernels built independently at `S d`. This file checks the ASSEMBLY that applies them: that the
# right canonical component is picked for each shape, that the declaration is not inferred from the
# container, and that a mismatch is an error rather than a plausible-looking reduction.
#
# The four declarations and the shapes they go with:
#
#     nothing          a single ToeplitzMatrixRepresentation
#     VectorField()    a 3-vector of them
#     TrueTensor()     a 3x3 of them
#     PseudoTensor()   a 3x3 of them, det(S) on top

const CHIRAL = H2Trees.LatticeSymmetry((2, 3, 1), (1, -1, 1))

# A distinguishable tensor per component, so a wrong component source cannot coincide with a right
# one. `n = 3` -> a 5x5x5 difference cube, the smallest that induces the full 48-element group.
_tensor(tag) = ComplexF64[tag + 1im * (a + 10b + 100c) for a in 1:5, b in 1:5, c in 1:5]
_rep(tag) = H2Factory.ToeplitzMatrixRepresentation(_tensor(tag))

const TSIZE = (5, 5, 5)
const PERM = H2Factory.toeplitzpermutation(TSIZE, CHIRAL)

@testset "scalar: the sample permutation is the whole action" begin
    value = _rep(0.0)
    out = H2Factory._reducedtranslation(value, PERM, CHIRAL, nothing)
    @test vec(collect(out.toeplitz)) == vec(_tensor(0.0))[PERM]
    # The canonical is SHARED, not copied, that sharing is the entire point of the reduction.
    @test H2Factory._reducedtranslation(value, PERM, CHIRAL, nothing).toeplitz.canonical ===
        value.toeplitz
end

@testset "VectorField: component i comes from perm[i], scaled by signs[i]" begin
    value = [_rep(Float64(i)) for i in 1:3]
    out = H2Factory._reducedtranslation(value, PERM, CHIRAL, H2Factory.VectorField())
    @test length(out) == 3
    for i in 1:3
        source, scale = H2Factory.componentsource(CHIRAL, i)
        @test vec(collect(out[i].toeplitz)) == scale .* vec(_tensor(Float64(source)))[PERM]
    end
    # NON-VACUITY: `CHIRAL` genuinely moves components, so an identity-component version differs.
    @test any(vec(collect(out[i].toeplitz)) != vec(_tensor(Float64(i)))[PERM] for i in 1:3)
end

@testset "TrueTensor and PseudoTensor differ by exactly det(S)" begin
    value = [_rep(Float64(i + 3j)) for i in 1:3, j in 1:3]
    truet = H2Factory._reducedtranslation(value, PERM, CHIRAL, H2Factory.TrueTensor())
    pseudo = H2Factory._reducedtranslation(value, PERM, CHIRAL, H2Factory.PseudoTensor())

    @test size(truet) == size(pseudo) == (3, 3)
    d = H2Trees.determinant(CHIRAL)
    @test d == -1                       # a reflection: the two declarations must disagree here
    for i in 1:3, j in 1:3
        @test vec(collect(pseudo[i, j].toeplitz)) == d .* vec(collect(truet[i, j].toeplitz))
        si, sj, scale = H2Factory.componentsource(CHIRAL, i, j)
        @test vec(collect(truet[i, j].toeplitz)) ==
            scale .* vec(_tensor(Float64(si + 3sj)))[PERM]
    end
end

@testset "the declaration is not inferred from the container shape" begin
    # THE WHOLE REASON THE DECLARATION EXISTS. A 3x3 coupling is a 3x3 coupling either way, so a
    # build that guessed from the shape would be right on half the group and O(1) wrong on the rest.
    matrix = [_rep(Float64(i + 3j)) for i in 1:3, j in 1:3]
    vector = [_rep(Float64(i)) for i in 1:3]

    @test_throws ErrorException H2Factory._reducedtranslation(matrix, PERM, CHIRAL, nothing)
    @test_throws ErrorException H2Factory._reducedtranslation(
        matrix, PERM, CHIRAL, H2Factory.VectorField()
    )
    @test_throws ErrorException H2Factory._reducedtranslation(
        vector, PERM, CHIRAL, H2Factory.TrueTensor()
    )
    @test_throws ErrorException H2Factory._reducedtranslation(
        _rep(0.0), PERM, CHIRAL, H2Factory.TrueTensor()
    )
end

@testset "an untagged collection is refused rather than silently unreduced" begin
    # An already-reduced collection reports no symmetry group. Reducing it again would produce a
    # collection that CLAIMS a reduction it did not perform, which is worse than an error.
    already = H2Factory.SymmetryReducedI2OTranslationCollection(
        Dict((1, 2) => _rep(0.0)),
        H2Factory.ToeplitzI2OTranslationTrait(),
        H2Factory.TranslationStats(1, 1, [1], [1], 0.0),
    )
    @test H2Factory.translationsymmetrygroup(already) === nothing
    @test_throws ErrorException H2Factory.symmetryreducedtoeplitzcollection(already)
end

@testset "permutations are shared between interactions in the same orbit" begin
    # NOT COSMETIC. A permutation is `prod(toeplitzsize)` Int32s, for a scalar coupling that is
    # BIGGER than the tensor it permutes. Built per interaction, the "reduction" costs more memory
    # than it saves: measured 1.4x bigger than unreduced on a real tree before this was cached.
    # So a regression here silently turns the feature into a pessimisation, and only object identity
    # can see it, the values would still be correct.
    canonical = _rep(0.0)
    a = H2Factory._reducedtranslation(canonical, PERM, CHIRAL, nothing)
    b = H2Factory._reducedtranslation(canonical, PERM, CHIRAL, nothing)
    @test a.toeplitz.permutation === b.toeplitz.permutation === PERM

    # And within one container the sample permutation is one object across all nine components.
    matrix = [_rep(Float64(i + 3j)) for i in 1:3, j in 1:3]
    out = H2Factory._reducedtranslation(matrix, PERM, CHIRAL, H2Factory.TrueTensor())
    @test all(o.toeplitz.permutation === PERM for o in out)
end

@testset "_toeplitztensorsize reads through every stored shape" begin
    @test H2Factory._toeplitztensorsize(_rep(0.0)) == TSIZE
    @test H2Factory._toeplitztensorsize([_rep(Float64(i)) for i in 1:3]) == TSIZE
    @test H2Factory._toeplitztensorsize([_rep(1.0) for i in 1:3, j in 1:3]) == TSIZE
end

end # module TestReducedToeplitz
