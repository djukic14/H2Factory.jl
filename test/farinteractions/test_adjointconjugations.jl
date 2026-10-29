module TestAdjointConjugations

using Test
using Random
using LinearAlgebra
using H2Factory

# THE CONJUGATED LEAF MULTIPLY, TESTED AS THE TWO LINEAR MAPS IT IS.
#
# An adjoint is the transpose with EVERY factor conjugated. The traversal conjugates the ones it owns
# -- `adjoint(transfer)` at the level-transfer boundary, `AdjointTranslation` at the coupling
# boundary -- and a map whose LEAF COLLECTIONS are complex supplies the remaining one by selecting
# `ConjugatedCollectionMul` from `leafmul(A, ::isAdjoint)`. `LinAlgebraMulFunctor`'s default is right
# only for real collections, which is what its own comment says.
#
# THIS FILE TESTS H2FACTORY'S KERNELS, NOT A MAP. Whether a given map SELECTS the functor, and whether
# a whole matvec then agrees with its exact matrix, is a property of that map and is checked where the
# map lives -- for the synthetic ones, in `H2FactoryTestUtils`' own suite, which is where a fake map's
# correctness belongs. Here the two kernels are pinned directly against the linear algebra they claim
# to compute, so a failure says which kernel and not merely "the adjoint is wrong".

@testset "Aggregation kernel computes conj(C) * x" begin
    rng = MersenneTwister(21)
    rank, ndofs = 5, 7
    C = randn(rng, ComplexF64, rank, ndofs)
    collection = H2Factory.MomentCollection1D{ComplexF64}((rank, ndofs))
    collection[1] .= C
    moment = H2Factory.Moment1D{ComplexF64}((rank,))
    x = randn(rng, ComplexF64, ndofs)

    H2Factory.applyconjugatedleafmul!(H2Factory.EmptyChannels(), moment, collection, x)
    @test H2Factory.wrapvector(moment, 1) ≈ conj(C) * x

    # NOT `C * conj(x)`, which is what `conjmul!` alone gives and what an earlier scheme used. The two
    # differ for complex `x`, and confusing them is the documented way to get this wrong.
    @test !isapprox(H2Factory.wrapvector(moment, 1), C * conj(x); rtol=1e-8)

    # The functor is a thin wrapper over the kernel and must agree with it.
    viafunctor = H2Factory.Moment1D{ComplexF64}((rank,))
    H2Factory.ConjugatedCollectionMul()(
        H2Factory.EmptyChannels(), viafunctor, collection, x
    )
    @test H2Factory.wrapvector(viafunctor, 1) == H2Factory.wrapvector(moment, 1)
end

@testset "Disaggregation kernel computes α * adjoint(C) * m + β * result" begin
    rng = MersenneTwister(22)
    rank, ndofs = 5, 7
    C = randn(rng, ComplexF64, rank, ndofs)
    collection = H2Factory.MomentCollection1D{ComplexF64}((rank, ndofs))
    collection[1] .= C
    moment = H2Factory.Moment1D{ComplexF64}((rank,))
    H2Factory.wrapvector(moment, 1) .= randn(rng, ComplexF64, rank)
    m = copy(H2Factory.wrapvector(moment, 1))

    α, β = 2.0 - 1.0im, 0.5 + 0.25im
    result = randn(rng, ComplexF64, ndofs)
    before = copy(result)
    H2Factory.applyconjugatedleafmul!(
        H2Factory.EmptyChannels(), result, collection, moment, α, β
    )
    @test result ≈ α * adjoint(C) * m + β * before

    # `adjoint`, not `transpose`: the two differ for a complex collection, and a kernel that used
    # `transpose` here would still satisfy every real-valued fixture.
    @test !isapprox(result, α * transpose(C) * m + β * before; rtol=1e-8)

    # `α` ARRIVES ALREADY CONJUGATED, from the map's own `α(A, ::isAdjoint)`. This kernel must NOT
    # conjugate it again -- doing so would undo the map's work and is invisible for a real `α`.
    result2 = copy(before)
    H2Factory.applyconjugatedleafmul!(
        H2Factory.EmptyChannels(), result2, collection, moment, conj(α), β
    )
    @test !isapprox(result2, result; rtol=1e-8)
end

@testset "β scales the pre-existing result once, not once per component" begin
    # THE SUBTLE ONE. With several components the kernel accumulates, and `β` must apply to the FIRST
    # component only -- later ones use `βnew = 1`. Applying it per component would scale `result` D
    # times, which is invisible whenever `β` is 0 or 1 (the traversal's usual values) and wrong for
    # any other.
    rng = MersenneTwister(23)
    D, rank, ndofs = 3, 4, 6
    collection = H2Factory.MomentCollection{D,ComplexF64}((rank, ndofs))
    moment = H2Factory.Moment{D,ComplexF64}((rank,))
    for i in 1:D
        collection[i] .= randn(rng, ComplexF64, rank, ndofs)
        H2Factory.wrapvector(moment, i) .= randn(rng, ComplexF64, rank)
    end

    α, β = 2.0 - 1.0im, 0.5 + 0.25im
    result = randn(rng, ComplexF64, ndofs)
    before = copy(result)
    H2Factory.applyconjugatedleafmul!(
        H2Factory.EmptyChannels(), result, collection, moment, α, β
    )

    expected = β * before
    for i in 1:D
        expected += α * adjoint(collection[i]) * H2Factory.wrapvector(moment, i)
    end
    @test result ≈ expected

    # Discriminating: the per-component-β version would give `β^D * before + ...`, which for this
    # `β` differs by far more than the tolerance.
    wrong = β^D * before
    for i in 1:D
        wrong += α * adjoint(collection[i]) * H2Factory.wrapvector(moment, i)
    end
    @test !isapprox(result, wrong; rtol=1e-8)
end

@testset "_componentα selects per component only for a vector α" begin
    # `α` is a scalar for some operators and a per-component vector for others; the kernels take both
    # through one helper so their bodies stay single.
    @test H2Factory._componentα(2.0 - 1.0im, 3) == 2.0 - 1.0im
    @test H2Factory._componentα([1.0, 2.0, 3.0], 2) == 2.0
end

# THE SELECTION ITSELF, not the kernels: `leafmul` must REFUSE the adjoint rather than answer it.
#
# The kernels above are worthless if nothing selects them. A shared `::AbstractMapTrait` default used
# to answer all three orientations with the plain matvec, so a map with complex leaf collections that
# never wrote an adjoint method silently got `LinAlgebraMulFunctor` and returned a well-formed wrong
# field. Forward and transpose genuinely have no such problem, which is why they keep a default and
# the adjoint does not.

# A core that declares nothing, standing in for a map whose author never considered the adjoint.
struct UndeclaredLeafMulCore <: H2Factory.AbstractH2Core{ComplexF64} end

# ...and one that declares, the way iFMM's real-collection cores now do.
struct DeclaredRealLeafMulCore <: H2Factory.AbstractH2Core{ComplexF64} end
function H2Factory.leafmul(::DeclaredRealLeafMulCore, ::H2Factory.isAdjoint)
    return H2Factory.LinAlgebraMulFunctor()
end

@testset "leafmul: the adjoint orientation is opt-in, not inherited" begin
    # Forward and transpose keep a shared default, because the plain matvec is correct there for
    # ANY leaf collection. If these start throwing, the narrowing went too wide.
    @test H2Factory.leafmul(UndeclaredLeafMulCore(), H2Factory.isForward()) isa
        H2Factory.LinAlgebraMulFunctor
    @test H2Factory.leafmul(UndeclaredLeafMulCore(), H2Factory.isTranspose()) isa
        H2Factory.LinAlgebraMulFunctor

    # NEGATIVE CONTROL, and the point of the whole change: the adjoint refuses a core that has not
    # opted in. Before the narrowing this returned `LinAlgebraMulFunctor()`, and this test could not
    # have been written because there was nothing to catch.
    @test_throws ErrorException H2Factory.leafmul(
        UndeclaredLeafMulCore(), H2Factory.isAdjoint()
    )

    # The message has to name the offending type and say what to write, or it is a MethodError with
    # extra steps.
    message = try
        H2Factory.leafmul(UndeclaredLeafMulCore(), H2Factory.isAdjoint())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("UndeclaredLeafMulCore", message)
    @test occursin("isAdjoint", message)
    @test occursin("LinAlgebraMulFunctor", message)

    # A core that HAS opted in gets exactly what it declared, so the refusal is about silence rather
    # than about the plain matvec being forbidden.
    @test H2Factory.leafmul(DeclaredRealLeafMulCore(), H2Factory.isAdjoint()) isa
        H2Factory.LinAlgebraMulFunctor
end

end # module TestAdjointConjugations
