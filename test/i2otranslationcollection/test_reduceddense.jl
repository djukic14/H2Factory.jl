module TestReducedDense

using Test
using LinearAlgebra
using H2Factory, H2Trees

# THE DENSE REDUCTION'S APPLY PATH: gather -> BLAS -> scatter, against the matrix it stands for.
#
# The claim is `out == (P M Qᵀ) in` computed WITHOUT ever forming `P M Qᵀ`. So every check here builds
# that matrix explicitly and compares, the explicit form is the oracle, and it is independently
# derived (a permutation-matrix product) rather than another call into the same code.

const CHIRAL = H2Trees.LatticeSymmetry((2, 3, 1), (1, -1, 1))
const B = 3
const N = B^3
const PERM = H2Factory.latticegridpermutation((B, B, B), CHIRAL)

# `P` with `(P v)[perm[i]] == v[i]`, matching `_scatter!`.
function permutationmatrix(permutation)
    P = zeros(Int, length(permutation), length(permutation))
    for i in eachindex(permutation)
        P[permutation[i], i] = 1
    end
    return P
end

const P = permutationmatrix(PERM)

_scratch(T, n) = (gathered=Vector{T}(undef, n), product=Vector{T}(undef, n))

# Moments, because the dense apply operates on a MOMENT and not a vector.
function _momentfrom(vectors)
    moment = H2Factory.Moment{length(vectors),ComplexF64}((length(first(vectors)),))
    for i in eachindex(vectors)
        H2Factory.wrapvector(moment, i) .= vectors[i]
    end
    return moment
end

_vectors(moment) = [copy(H2Factory.wrapvector(moment, i)) for i in eachindex(moment)]

@testset "P is a genuine permutation matrix and CHIRAL is not self-inverse" begin
    @test P * transpose(P) == I
    # NON-VACUITY for everything below: on a self-inverse element a swapped convention is invisible.
    @test PERM !=
        H2Factory.latticegridpermutation((B, B, B), H2Trees.inversesymmetry(CHIRAL))
end

@testset "forward: gather/BLAS/scatter equals P M Pᵀ" begin
    M = randn(ComplexF64, N, N)
    x = randn(ComplexF64, N)
    translation = H2Factory.PermutedDenseTranslation(M, PERM)

    out = zeros(ComplexF64, N)
    H2Factory._permuteddensemul!(out, x, translation, _scratch(ComplexF64, N), identity)
    @test out ≈ P * M * transpose(P) * x

    # The canonical is SHARED, never permuted, that is the entire point.
    @test translation.canonical === M
end

@testset "transpose and adjoint reuse the SAME permutation" begin
    M = randn(ComplexF64, N, N)
    x = randn(ComplexF64, N)
    translation = H2Factory.PermutedDenseTranslation(M, PERM)
    full = P * M * transpose(P)

    out = zeros(ComplexF64, N)
    H2Factory._permuteddensemul!(out, x, translation, _scratch(ComplexF64, N), transpose)
    @test out ≈ transpose(full) * x

    out = zeros(ComplexF64, N)
    H2Factory._permuteddensemul!(out, x, translation, _scratch(ComplexF64, N), adjoint)
    @test out ≈ adjoint(full) * x

    # M is non-symmetric and complex, so the three modes are genuinely different operators, a
    # kernel that confused them would otherwise pass.
    @test norm(full - transpose(full)) / norm(full) > 0.1
    @test norm(full - adjoint(full)) / norm(full) > 0.1
end

@testset "RECTANGULAR: rows and columns carry DIFFERENT permutations" begin
    # The square case cannot tell a row/column mix-up from the truth, and `_sharedi2ogrids` is false
    # for the dense route, a Petrov coupling really can have different grids per side. Here the two
    # permutations have different LENGTHS, so exchanging them is a `DimensionMismatch` rather than a
    # wrong number; what this pins is that forward gathers over COLUMNS and scatters over ROWS, and
    # that transpose does the opposite.
    brow, bcolumn = 3, 2
    nrow, ncolumn = brow^3, bcolumn^3
    rowpermutation = H2Factory.latticegridpermutation((brow, brow, brow), CHIRAL)
    columnpermutation = H2Factory.latticegridpermutation(
        (bcolumn, bcolumn, bcolumn), CHIRAL
    )
    Prow = permutationmatrix(rowpermutation)
    Pcolumn = permutationmatrix(columnpermutation)

    M = randn(ComplexF64, nrow, ncolumn)
    translation = H2Factory.PermutedDenseTranslation(M, rowpermutation, columnpermutation)
    full = Prow * M * transpose(Pcolumn)
    scratch = _scratch(ComplexF64, max(nrow, ncolumn))

    x = randn(ComplexF64, ncolumn)
    out = zeros(ComplexF64, nrow)
    H2Factory._permuteddensemul!(out, x, translation, scratch, identity)
    @test out ≈ full * x

    y = randn(ComplexF64, nrow)
    out = zeros(ComplexF64, ncolumn)
    H2Factory._permuteddensemul!(out, y, translation, scratch, transpose)
    @test out ≈ transpose(full) * y

    out = zeros(ComplexF64, ncolumn)
    H2Factory._permuteddensemul!(out, y, translation, scratch, adjoint)
    @test out ≈ adjoint(full) * y
end

@testset "the alpha/beta form accumulates rather than overwrites" begin
    M = randn(ComplexF64, N, N)
    x = randn(ComplexF64, N)
    y = randn(ComplexF64, N)
    translation = H2Factory.PermutedDenseTranslation(M, PERM)
    full = P * M * transpose(P)

    out = copy(y)
    H2Factory._permuteddensemul!(
        out, x, translation, _scratch(ComplexF64, N), identity, true, true
    )
    @test out ≈ full * x + y

    # beta = false must overwrite, including where `out` held junk.
    out = copy(y)
    H2Factory._permuteddensemul!(
        out, x, translation, _scratch(ComplexF64, N), identity, true, false
    )
    @test out ≈ full * x
end

@testset "an UNTAGGED entry goes straight to the unreduced mul!" begin
    # A GUARD, not the common path: measured on a sphere tree, every interaction carries a tag (a
    # representative's being the identity element rather than no element). This pins the mixed-entry
    # case anyway, since the builder can still produce it and a `MethodError` in the traversal is a
    # poor way to find that out.
    #
    # Driven at the MOMENT level, which is where production handles it, the earlier vector-level
    # methods were dead once that moved, and a test aimed at a bypassed layer proves nothing about
    # the path a traversal actually takes.
    M = randn(ComplexF64, N, N)
    scratch = _scratch(ComplexF64, N)
    input = _momentfrom([randn(ComplexF64, N)])

    for (operation, expected) in
        ((identity, M), (transpose, transpose(M)), (adjoint, adjoint(M)))
        out = _momentfrom([zeros(ComplexF64, N)])
        H2Factory._applypermuteddense!(out, input, M, scratch, operation)
        @test only(_vectors(out)) ≈ expected * only(_vectors(input))
    end

    # The scratch is genuinely untouched: no gather or scatter is paid for a representative.
    fill!(scratch.gathered, 0)
    out = _momentfrom([zeros(ComplexF64, N)])
    H2Factory._applypermuteddense!(out, input, M, scratch, identity)
    @test all(iszero, scratch.gathered)

    # The alpha/beta form accumulates through the same route.
    existing = randn(ComplexF64, N)
    out = _momentfrom([existing])
    H2Factory._applypermuteddense!(out, input, M, scratch, identity, true, true)
    @test only(_vectors(out)) ≈ M * only(_vectors(input)) + existing
end

@testset "shape guards" begin
    M = randn(ComplexF64, N, N)
    @test_throws DimensionMismatch H2Factory.PermutedDenseTranslation(M, PERM[1:(N - 1)])
    @test_throws DimensionMismatch H2Factory.PermutedDenseTranslation(
        randn(ComplexF64, N, N - 1), PERM
    )
end

@testset "it is NOT an AbstractArray" begin
    # Deliberate: an array would reach `LinearAlgebra.mul!` and silently take the generic,
    # non-BLAS path, correct results at a cost that erases the reason for the feature.
    translation = H2Factory.PermutedDenseTranslation(randn(ComplexF64, N, N), PERM)
    @test !(translation isa AbstractArray)
    @test size(translation) == (N, N)
    @test eltype(translation) == ComplexF64
end

# THE 3x3 BLOCK REDUCTION, AGAINST THE UNREDUCED PATH ITSELF.
#
# The oracle here is the PRODUCTION `mul!` for a moment and a `Matrix{Matrix}` coupling, the exact
# call the unreduced `DenseTranslation` kernel makes. So this asserts the reduced kernel is
# indistinguishable from the thing it replaces, rather than agreeing with a second implementation of
# the same idea.

# The 3x3 container the reduced entry stands for: block (a,b) is the canonical block (pa,pb), signed.
function _materializeblocks(canonical, symmetry, permutation, kind)
    full = [zeros(ComplexF64, size(canonical[1, 1])) for _ in 1:3, _ in 1:3]
    for a in 1:3, b in 1:3
        sourcea, sourceb, scale = H2Factory.componentsource(kind, symmetry, a, b)
        source = canonical[sourcea, sourceb]
        for j in axes(source, 2), i in axes(source, 1)
            full[a, b][permutation[i], permutation[j]] = scale * source[i, j]
        end
    end
    return full
end

@testset "3x3 blocks: $kindname agrees with the unreduced mul!" for (kindname, kind) in (
    ("TrueTensor", H2Factory.TrueTensor()), ("PseudoTensor", H2Factory.PseudoTensor())
)
    canonical = [randn(ComplexF64, N, N) for _ in 1:3, _ in 1:3]
    translation = H2Factory._reduceddensetranslation(canonical, PERM, PERM, CHIRAL, kind)
    full = _materializeblocks(canonical, CHIRAL, PERM, kind)
    scratch = _scratch(ComplexF64, N)

    input = _momentfrom([randn(ComplexF64, N) for _ in 1:3])

    for (name, operation) in
        (("forward", identity), ("transpose", transpose), ("adjoint", adjoint))
        reduced = _momentfrom([zeros(ComplexF64, N) for _ in 1:3])
        oracle = _momentfrom([zeros(ComplexF64, N) for _ in 1:3])
        H2Factory._applypermuteddense!(reduced, input, translation, scratch, operation)
        LinearAlgebra.mul!(oracle, operation(full), input)
        @test _vectors(reduced) ≈ _vectors(oracle)
    end

    # THE BLOCKS ARE SHARED, never copied, the entire point, and what a negated-copy
    # implementation would silently give up.
    for a in 1:3, b in 1:3
        sourcea, sourceb, _ = H2Factory.componentsource(kind, CHIRAL, a, b)
        @test translation.blocks[a, b] === canonical[sourcea, sourceb]
    end

    # NON-VACUITY: the signs are load-bearing. `CHIRAL` is a reflection, so the two parity rules
    # differ, and a kernel ignoring the rule would pass under either.
    other =
        kind === H2Factory.TrueTensor() ? H2Factory.PseudoTensor() : H2Factory.TrueTensor()
    wrong = _materializeblocks(canonical, CHIRAL, PERM, other)
    @test !isapprox(full[1, 1], wrong[1, 1])
end

@testset "3x3 blocks: the alpha/beta form accumulates" begin
    canonical = [randn(ComplexF64, N, N) for _ in 1:3, _ in 1:3]
    kind = H2Factory.TrueTensor()
    translation = H2Factory._reduceddensetranslation(canonical, PERM, PERM, CHIRAL, kind)
    full = _materializeblocks(canonical, CHIRAL, PERM, kind)
    scratch = _scratch(ComplexF64, N)

    input = _momentfrom([randn(ComplexF64, N) for _ in 1:3])
    existing = [randn(ComplexF64, N) for _ in 1:3]

    reduced, oracle = _momentfrom(existing), _momentfrom(existing)
    H2Factory._applypermuteddense!(
        reduced, input, translation, scratch, identity, true, true
    )
    LinearAlgebra.mul!(oracle, full, input, true, true)
    @test _vectors(reduced) ≈ _vectors(oracle)
end

@testset "VectorField is refused on the dense route, and the message says why" begin
    # NOT merely unwritten: the operators declaring `VectorField` have `trialmoment != testmoment`,
    # so they cannot form a Galerkin map at all, and the Petrov dense route is refused separately.
    # A branch here would be surface no configuration can reach.
    blocks = [randn(ComplexF64, N, N) for _ in 1:3]
    @test_throws ErrorException H2Factory._reduceddensetranslation(
        blocks, PERM, PERM, CHIRAL, H2Factory.VectorField()
    )
    @test_throws "cannot form a Galerkin" H2Factory._reduceddensetranslation(
        blocks, PERM, PERM, CHIRAL, H2Factory.VectorField()
    )

    # And a shape/declaration MISMATCH is still an error, not a fallback: a 3x3 coupling declaring
    # `nothing` would otherwise be reduced with the sample permutation alone, which is exactly the
    # silently-wrong-on-half-the-group outcome the parity rules exist to prevent.
    @test_throws ErrorException H2Factory._reduceddensetranslation(
        [randn(ComplexF64, N, N) for _ in 1:3, _ in 1:3], PERM, PERM, CHIRAL, nothing
    )
end

end # module TestReducedDense
