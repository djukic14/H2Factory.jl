module TestSideOwnership

using Test
using Random
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils

# Part E Phase 5 PREREQUISITE.
#
# Every other harness test drives the fakes NUMERICALLY: build a map, multiply, compare against an
# independently assembled dense matrix. That is exactly the check a side-ownership bug survives. A
# Petrov map whose test and trial sides carry the SAME transfer object produces the same numbers
# whether or not the orientation flip is applied, so the flip is unobservable and a traversal that
# silently applies the forward (trial) transfer in transposed mode still matches its dense reference
# to machine precision.
#
# That is not hypothetical. It has happened twice in this refactor, and both times three green suites
# plus the factorization oracle missed it:
#
#   1. `I2ITranslatorFunctor`/`O2OTranslatorFunctor` lost the side flip by taking the MAP and
#      re-resolving the transfer at the call site (against the already-unwrapped forward map).
#   2. `toeplitzcompatible` was written as `==`, which is not structural for buffered B-splines --
#      caught only when a test finally used a type whose two sides are DISTINCT OBJECTS.
#
# Both share one shape: a per-side distinction that collapses when both sides resolve to one object.
#
# So these tests are deliberately STRUCTURAL, not numerical. They assert that each stage reaches for
# the correct SIDE, using per-side sentinels that are distinguishable by identity. A numerical
# assertion cannot express this; only identity can.
#
# The contract under test (H2Factory `maps/accessors.jl`). Since the level-transfer refactor,
# disaggregation is "O2O is no longer semantic": it resolves the disaggregation SIDE's I2I family and
# applies it through an explicit transpose/adjoint (`disaggregationi2itranslator` +
# `disaggregationdirection`), so the FUNCTOR carries an I2I translator, not a materialized O2O.
#
#     aggregation    forward -> TRIAL i2i (forward)      transposed/adjoint -> TEST  i2i (forward)
#     disaggregation forward -> TEST  i2i (transpose)    transposed        -> TRIAL i2i (transpose)
#                                                        adjoint           -> TRIAL i2i (adjoint)
#
# The side flip is trial<->test; the O2O action (`O2O == transpose(I2I)`) is reconstructed at the
# transfer boundary by the direction, not by a separate materialized O2O object. The materialized-O2O
# accessors (`testo2otranslator` etc.) still exist as a cache but are OFF the traversal path.

# Sentinel transfers. `sidesentinel` carries a label and otherwise behaves as the identity transfer,
# so a map built from four distinct sentinels still multiplies correctly -- letting the same fake
# serve both the structural assertions here and a numerical sanity check.
struct SideSentinel{F}
    label::Symbol
    inner::F
end
label(s::SideSentinel) = s.label

sidesentinel(lbl::Symbol) = SideSentinel(lbl, identitytranslator())

# Delegate every translator call form to the wrapped identity translator.
(s::SideSentinel)(args...) = s.inner(args...)

# Seeded. These assertions are structural -- they do not depend on the geometry at all -- but a
# deterministic tree means a future failure is reproducible on the spot rather than needing the RNG
# state recovered first.
const SEED = 20260720

@testset "Petrov fakes carry genuinely side-owned transfers" begin
    rng = MersenneTwister(SEED)
    testpoints = [SVector(0.1randn(rng), 0.1randn(rng), 0.1randn(rng)) for _ in 1:64]
    trialpoints = [SVector(0.1randn(rng), 0.1randn(rng), 0.1randn(rng)) for _ in 1:64]

    A = fakepetrovmap(
        testpoints,
        trialpoints;
        T=ComplexF64,
        testi2i=sidesentinel(:testi2i),
        triali2i=sidesentinel(:triali2i),
        testo2o=sidesentinel(:testo2o),
        trialo2o=sidesentinel(:trialo2o),
    )

    # 1. The four sides must be DISTINCT OBJECTS. If this collapses, every assertion below is
    #    vacuous -- it would pass just as well against a map with one shared transfer.
    @test label(H2Factory.triali2itranslator(A)) === :triali2i
    @test label(H2Factory.testi2itranslator(A)) === :testi2i
    @test label(H2Factory.trialo2otranslator(A)) === :trialo2o
    @test label(H2Factory.testo2otranslator(A)) === :testo2o
    @test H2Factory.triali2itranslator(A) !== H2Factory.testi2itranslator(A)
    @test H2Factory.trialo2otranslator(A) !== H2Factory.testo2otranslator(A)

    # 2. Orientation resolution: aggregation is trial-side forward, test-side transposed;
    #    disaggregation mirrors it.
    @test label(H2Factory.i2itranslator(A, H2Factory.isForward())) === :triali2i
    @test label(H2Factory.i2itranslator(A, H2Factory.isTranspose())) === :testi2i
    @test label(H2Factory.i2itranslator(A, H2Factory.isAdjoint())) === :testi2i
    @test label(H2Factory.o2otranslator(A, H2Factory.isForward())) === :testo2o
    @test label(H2Factory.o2otranslator(A, H2Factory.isTranspose())) === :trialo2o
    @test label(H2Factory.o2otranslator(A, H2Factory.isAdjoint())) === :trialo2o

    # 3. THE ASSERTION THAT WOULD HAVE CAUGHT THE FUNCTOR BUG. The traversal does not call the
    #    accessors above -- it calls the FUNCTOR accessors, which resolve the translator and hand the
    #    functor the resolved object. The earlier bug passed the MAP instead, so the call site
    #    re-resolved against the unwrapped forward map and silently used the forward side in
    #    transposed mode. Assertions 1-2 above stayed green throughout. Only reaching THROUGH the
    #    wrapper, as the traversal does, exposes it.
    #
    #    The I2I functor carries the aggregation side's i2i; the O2O (disaggregation) functor now
    #    carries the disaggregation side's I2I too (O2O is reconstructed by the direction), so its
    #    labels are the i2i sentinels, mirrored trial<->test, NOT the o2o sentinels.
    @test label(H2Factory.i2itranslatorfunctor(A).translator) === :triali2i
    @test label(H2Factory.i2itranslatorfunctor(transpose(A)).translator) === :testi2i
    @test label(H2Factory.i2itranslatorfunctor(adjoint(A)).translator) === :testi2i
    @test label(H2Factory.o2otranslatorfunctor(A).translator) === :testi2i
    @test label(H2Factory.o2otranslatorfunctor(transpose(A)).translator) === :triali2i
    @test label(H2Factory.o2otranslatorfunctor(adjoint(A)).translator) === :triali2i

    # The disaggregation functor also carries the explicit direction: transpose for forward/transpose
    # maps, adjoint for an adjoint map (this is what reconstructs O2O == transpose/adjoint(I2I)).
    @test H2Factory.o2otranslatorfunctor(A).direction isa H2Factory.TransposeTranslation
    @test H2Factory.o2otranslatorfunctor(transpose(A)).direction isa
        H2Factory.TransposeTranslation
    @test H2Factory.o2otranslatorfunctor(adjoint(A)).direction isa
        H2Factory.AdjointTranslation

    # 4. Non-vacuity: the sentinels must be reachable through a REAL wrapped map, not only through a
    #    bare one. `transpose(A)` is what the traversal actually receives.
    @test H2Factory.i2itranslatorfunctor(transpose(A)).translator !==
        H2Factory.i2itranslatorfunctor(A).translator
end

@testset "Galerkin fakes collapse the sides, because that IS the mathematics" begin
    # A Galerkin map has ONE tree and one space, so test and trial genuinely coincide -- collapsing
    # is correct here, not a limitation. Asserted explicitly so the collapse stays a deliberate
    # mathematical statement rather than an accident of the fake's field layout.
    # Note `fakegalerkinmap` takes a TREE (one tree is the whole point of a Galerkin map), whereas
    # `fakepetrovmap` takes two POINT SETS and builds the `BlockTree` itself.
    rng = MersenneTwister(SEED)
    points = [SVector(0.1randn(rng), 0.1randn(rng), 0.1randn(rng)) for _ in 1:64]
    tree = H2Trees.buildtree(
        points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.05, minvalues=0, root=1)
    )
    A = fakegalerkinmap(tree; T=ComplexF64, i2i=sidesentinel(:shared))

    @test H2Factory.triali2itranslator(A) === H2Factory.testi2itranslator(A)
    @test label(H2Factory.i2itranslator(A, H2Factory.isForward())) === :shared
    @test label(H2Factory.i2itranslator(A, H2Factory.isTranspose())) === :shared
end

end # module TestSideOwnership
