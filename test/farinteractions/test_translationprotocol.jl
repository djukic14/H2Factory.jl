module TestTranslationProtocol

using Test
using LinearAlgebra
using StaticArrays
using H2Factory
using H2Trees
using H2FactoryTestUtils

# Unit tests for the translation-application protocol: `applycoupling!` dispatches on translation
# `direction` (Forward applies the coupling, Transpose its transpose, Adjoint its adjoint) and
# `accumulation` mode (Overwrite vs Accumulate), taking a `TranslationContext`. Plus the
# trait->direction mapping and the
# direction-parametrized `I2OTranslatorFunctor`. (Integration coverage: every `farmul!` in the
# component harnesses routes through this protocol.)
#
# These used to assert `applycoupling! == i2otranslation!/o2itranslation!/adjointo2itranslation!`, the
# old direction-named primitives it delegated to. Those are gone (#180 collapsed the two protocols), so
# there is no second side left to compare against -- and comparing two protocols never showed either
# was RIGHT, only that they agreed. Each direction is now checked against the operator it is defined to
# apply, written out explicitly.

@testset "translationdirection(maptrait)" begin
    @test H2Factory.translationdirection(H2Factory.isForward()) ===
        H2Factory.ForwardTranslation()
    @test H2Factory.translationdirection(H2Factory.isTranspose()) ===
        H2Factory.TransposeTranslation()
    @test H2Factory.translationdirection(H2Factory.isAdjoint()) ===
        H2Factory.AdjointTranslation()
end

@testset "applycoupling! applies the right operator for every direction x accumulation" begin
    # A `ScalarTranslator` coupling: forward and transpose apply `c` (a scalar is its own transpose),
    # the adjoint applies `conj(c)`. `c` is deliberately non-real so a dropped or misplaced
    # conjugation shows up numerically instead of cancelling.
    c = 2.0 - 1.0im
    t = scalartranslator(c)
    mkm(v) = (m=H2Factory.Moment1D{ComplexF64}((3,)); m[1].=v; m)
    x = randn(ComplexF64, 3)
    src = mkm(x)
    ctx = H2Factory.TranslationContext(H2Factory.SerialScheduler(), nothing)

    apply(direction, accumulation, initial) = begin
        out = mkm(copy(initial))
        H2Factory.applycoupling!(out, src, t, 3, 8, direction, accumulation, ctx)
        out[1]
    end

    zero3 = zeros(ComplexF64, 3)
    pre = randn(ComplexF64, 3)

    # Overwrite writes the operator's action; Accumulate adds it to what was already there.
    for (direction, expected) in (
        (H2Factory.ForwardTranslation(), c .* x),
        (H2Factory.TransposeTranslation(), c .* x),
        (H2Factory.AdjointTranslation(), conj(c) .* x),
    )
        @test apply(direction, H2Factory.Overwrite(), zero3) ≈ expected
        @test apply(direction, H2Factory.Accumulate(), pre) ≈ pre .+ expected
    end

    # Negative controls, so the three directions are not passing by coincidence. The adjoint must
    # genuinely differ from the transpose (it conjugates the coupling OPERATOR), and accumulating
    # into a nonzero target must genuinely differ from overwriting it.
    @test !(
        apply(H2Factory.AdjointTranslation(), H2Factory.Overwrite(), zero3) ≈
        apply(H2Factory.TransposeTranslation(), H2Factory.Overwrite(), zero3)
    )
    @test !(
        apply(H2Factory.ForwardTranslation(), H2Factory.Accumulate(), pre) ≈
        apply(H2Factory.ForwardTranslation(), H2Factory.Overwrite(), pre)
    )
end

@testset "applycoupling! is orchestration: exactly one method, and it is the guard" begin
    # WHAT THIS DOES AND DOES NOT CLAIM. Julia cannot make the bypass impossible: anyone may add a
    # more specific `applycoupling!` method at a concrete collection type, and it would shadow the
    # guarded one exactly as the old direction-named primitives did (that is how iFMM's B-spline
    # collection came to need six hand-written `assertnosymmetry` calls, #172). What is enforceable is
    # that nobody DOES, and that is what is pinned here and in the downstream extension-surface
    # guards: `applycoupling!` carries ONE method, in H2Factory, full stop.
    #
    # `H2FactoryTestUtils` is loaded in this environment, so this covers the mocks too -- they are the
    # most likely place for a convenience override to appear, since they have the least to lose from
    # skipping a guard they believe cannot fire.
    all = collect(methods(H2Factory.applycoupling!))
    @test length(all) == 1
    @test only(all).module === H2Factory

    # And that one method is the GUARD: it dispatches on the coupling supertype, so every coupling
    # collection -- production, dummy, or mock -- reaches `assertnosymmetry` through it. Typed on
    # `AbstractI2OCouplingCollection` rather than `AbstractI2OTranslationCollection`, which is the
    # widening that closed the hole: the narrower type left everything outside the
    # translation-collection branch free to implement the entry, and Dummy and the mocks did.
    @test only(all).sig.parameters[4] === H2Factory.AbstractI2OCouplingCollection

    # The kernel is a genuinely different generic with real methods behind it, so "implement the
    # kernel" is an instruction that can actually be followed.
    @test H2Factory.applycouplingkernel! !== H2Factory.applycoupling!
    @test length(methods(H2Factory.applycouplingkernel!)) > 10

    # `applycoupling!` is NOT declared in the extension namespace, which is what stops it being
    # offered as a hook in the first place. `applycouplingkernel!` is.
    @test !isdefined(H2Factory.Interfaces, :applycoupling!)
    @test isdefined(H2Factory.Interfaces, :applycouplingkernel!)
    @test H2Factory.applycouplingkernel! === H2Factory.Interfaces.applycouplingkernel!
end

# A coupling collection that declares neither answer. The negative control for the opt-in rule below:
# without it, "these types declare" is a statement with nothing to contrast against.
struct UndeclaredCoupling <: H2Factory.AbstractI2OCouplingCollection end

@testset "every coupling collection declares both answers the entry asks for" begin
    # #168's opt-in rule, now covering the whole `AbstractI2OCouplingCollection` hierarchy rather than
    # only its translation-collection branch. The entry asks each collection whether unconsumed
    # symmetry tags remain and what its storage shape is; neither has a permissive default, so a type
    # that forgets fails loudly instead of being waved through.
    #
    # Checked with `which` rather than by calling: calling `assertnosymmetry` and getting `nothing`
    # passes whether the method is the type's own or an inherited permissive fallback, which is the
    # entire distinction.
    abstractmethod = which(
        H2Factory.assertnosymmetry, Tuple{H2Factory.AbstractI2OCouplingCollection}
    )
    for collection in (
        H2Factory.DummyI2OTranslationCollection(),
        identitytranslator(),
        scalartranslator(2.0 - 1.0im),
        spytranslator(scalartranslator(2.0 - 1.0im)),
    )
        @test which(H2Factory.assertnosymmetry, Tuple{typeof(collection)}) !==
            abstractmethod
        @test H2Factory.assertnosymmetry(collection) === nothing
        @test H2Factory.i2otrait(collection) isa H2Factory.AbstractI2OTranslationTrait

        # And each is a declared coupling collection, so it composes into an `I2OTranslation`. This is
        # what the honest re-parenting bought: the mocks reach the protocol as a production collection
        # does, rather than through a broad fallback that treated any object as its own coupling.
        @test collection isa H2Factory.AbstractI2OCouplingCollection
        @test H2Factory.coupling(H2Factory.I2OTranslation(collection)) === collection
    end

    # The negative control: the abstract method REFUSES, so the declarations above are load-bearing
    # rather than decorative. A coupling collection that declares nothing reaches this, and reaches it
    # through the entry as well as directly -- the second assertion is what says the guard is actually
    # wired into the application path rather than merely present.
    @test_throws "assertnosymmetry is not declared" H2Factory.assertnosymmetry(
        UndeclaredCoupling()
    )
    @test_throws "assertnosymmetry is not declared" H2Factory.applycoupling!(
        nothing,
        nothing,
        UndeclaredCoupling(),
        3,
        8,
        H2Factory.ForwardTranslation(),
        H2Factory.Overwrite(),
        H2Factory.TranslationContext(H2Factory.SerialScheduler(), nothing),
    )
end

@testset "DiagonalTranslation adjoint: transform=conj, no per-translation coefficient copy" begin
    # A real DiagonalTranslation `I2OTranslationCollection` (not a mock translator), built via the
    # compatibility constructor from a hand-assembled dict -- exercises the actual
    # `elementwisemul!(...; transform=conj)` path the adjoint diagonal kernel routes through, since
    # `ConjugatedCoefficients` was removed (Part E cleanup).
    function runadjoint(n, accumulation)
        recv, transl = 3, 8
        coeffs = randn(ComplexF64, n)
        tfs = H2Factory.I2OTranslationCollection(
            Dict((transl, recv) => coeffs), H2Factory.DiagonalTranslation()
        )
        mkm(v) = (m=H2Factory.Moment1D{ComplexF64}((n,)); m[1].=v; m)
        src = mkm(randn(ComplexF64, n))
        out = mkm(randn(ComplexF64, n))
        pre = copy(out[1])

        args = (
            out,
            src,
            tfs,
            recv,
            transl,
            H2Factory.AdjointTranslation(),
            accumulation,
            H2Factory.TranslationContext(H2Factory.SerialScheduler(), nothing),
        )
        H2Factory.applycoupling!(args...)
        if accumulation === H2Factory.Overwrite()
            @test out[1] ≈ conj.(coeffs) .* src[1]
        else
            @test out[1] ≈ pre .+ conj.(coeffs) .* src[1]
        end

        # Warm up, then measure a repeat call. A `conj.(coefficients)` copy would allocate O(n);
        # this must not (an O(1) wrapper, like the removed `ConjugatedCoefficients`, would have
        # passed this check too -- the property actually being pinned is "no coefficient-sized
        # copy", which is what `transform=conj` avoids by conjugating inline per element instead).
        H2Factory.applycoupling!(args...)
        return @allocated H2Factory.applycoupling!(args...)
    end

    # Overwrite form.
    runadjoint(4, H2Factory.Overwrite())
    a_small = runadjoint(4, H2Factory.Overwrite())
    a_large = runadjoint(4_000, H2Factory.Overwrite())
    @test a_small == a_large   # allocation independent of the coefficient array size -> no copy

    # Accumulate form.
    runadjoint(4, H2Factory.Accumulate())
    aacc_small = runadjoint(4, H2Factory.Accumulate())
    aacc_large = runadjoint(4_000, H2Factory.Accumulate())
    @test aacc_small == aacc_large   # same: independent of size -> no copy
end

@testset "i2otranslatorfunctor direction tracks maptrait" begin
    tree = H2Trees.buildtree(
        [SVector(x, 0.0, 0.0) for x in (0.0, 0.1, 5.0, 5.1, 40.0, 40.1, 45.0, 45.1)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=2.0, minvalues=0, root=1),
    )
    A = fakegalerkinmap(tree; T=ComplexF64, i2o=scalartranslator(1.0 + 0.0im))
    @test H2Factory.i2otranslatorfunctor(A) isa
        H2Factory.I2OTranslatorFunctor{<:Any,H2Factory.ForwardTranslation}
    @test H2Factory.i2otranslatorfunctor(transpose(A)) isa
        H2Factory.I2OTranslatorFunctor{<:Any,H2Factory.TransposeTranslation}
    @test H2Factory.i2otranslatorfunctor(adjoint(A)) isa
        H2Factory.I2OTranslatorFunctor{<:Any,H2Factory.AdjointTranslation}
    # the 1-arg compatibility constructor defaults to the forward direction
    @test H2Factory.I2OTranslatorFunctor(A) isa
        H2Factory.I2OTranslatorFunctor{<:Any,H2Factory.ForwardTranslation}
end

@testset "translation functors take explicit accumulation modes" begin
    scheduler = H2Factory.SerialScheduler()
    functors = (
        H2Factory.I2ITranslatorFunctor(:translator),
        H2Factory.I2OTranslatorFunctor(:map),
        H2Factory.O2OTranslatorFunctor(:translator, H2Factory.ForwardTranslation()),
    )

    for f in functors
        args = (:out, :in, :tree, 1, 2)
        @test applicable(f, args..., scheduler, nothing)
        @test applicable(f, args..., H2Factory.Overwrite(), scheduler, nothing)
        @test applicable(f, args..., H2Factory.Accumulate(), scheduler, nothing)
        @test !applicable(f, args..., true, true, scheduler, nothing)
    end
end

end # module TestTranslationProtocol
