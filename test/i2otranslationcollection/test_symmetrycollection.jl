module TestSymmetryI2OCollection

using Test
using StaticArrays
using Random: Xoshiro
using H2Factory, H2Trees
using OhMyThreads: SerialScheduler

# Records the translation DIRECTION itself, so the stored value is literally the canonical offset
# and the symmetry can be checked against it without a translation operator in the way.
struct DirectionRecordingTranslator end
Base.eltype(::DirectionRecordingTranslator) = SVector{3,Float64}
(::DirectionRecordingTranslator)(receiving, translating, direction) = direction

@testset "Symmetry-reduced I2O translations" begin
    rng = Xoshiro(7)
    points = [SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) for _ in 1:2000]
    tree = H2Trees.TwoNTree(
        points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.3, minvalues=0)
    )
    plans = H2Trees.buildplans(tree)
    plan = H2Trees.translatingplan(plans.trialaggregationplan, plans.testdisaggregationplan)

    plaincollection, _ = H2Factory.I2OTranslationCollection(
        tree, plan, H2Trees.DirectionInvariancePerLevel(), DirectionRecordingTranslator()
    )
    plainstats = H2Factory.translationstats(plaincollection)

    # A symmetry-free collection reports no symmetry, and costs nothing for it.
    @test H2Factory.translationsymmetrygroup(plaincollection) === nothing
    @test H2Factory.translationsymmetry(plaincollection, 1, 2) === nothing

    for policy in (
        H2Trees.OppositeSymmetry(),
        H2Trees.AxisPreservingSymmetry(3),
        H2Trees.FullLatticeSymmetry(),
    )
        collection, _ = H2Factory.I2OTranslationCollection(
            tree,
            plan,
            H2Trees.SymmetryDirectionInvariancePerLevel(policy),
            DirectionRecordingTranslator(),
        )
        stats = H2Factory.translationstats(collection)

        # Same interactions, fewer stored translations.
        @test stats.npairs == plainstats.npairs
        @test stats.nunique < plainstats.nunique
        @test H2Factory.translationsymmetrygroup(collection) !== nothing

        # THE CONTRACT, on every pair rather than a sample: the recorded symmetry carries the
        # stored (canonical) direction onto the one this pair actually needs. This is what
        # `pairtranslationstore` would silently lose -- it keys by (receiving, translating) and
        # drops the translationID, so without the side table the tag has nowhere to live.
        allrecover = true
        for ((receiving, translating), canonical) in collection.dict
            h = H2Trees.halfsize(tree, receiving)
            actual = H2Trees.center(tree, receiving) - H2Trees.center(tree, translating)
            symmetry = H2Factory.translationsymmetry(collection, receiving, translating)
            q = ntuple(i -> round(Int, canonical[i] / h), 3)
            qactual = ntuple(i -> round(Int, actual[i] / h), 3)
            if H2Trees.applysymmetry(symmetry, q) != qactual
                allrecover = false
                break
            end
        end
        @test allrecover

        # Applying a symmetry-reduced collection through the plain path would use the ORBIT's
        # translation in place of this pair's own -- wrong for every non-identity symmetry, and
        # wrong quietly, since the shapes match and the result is a plausible field. It refuses.
        @test_throws "symmetry-reduced" H2Factory.assertnosymmetry(collection)

        # EVERY direction and accumulation mode, not just the accessor and not just the forward one.
        # Accumulate is the mode the traversal actually calls, and it was unguarded when the guard was
        # first added -- guarding only the other left the silent path open while the code read as
        # protected. The transpose and adjoint directions were unguarded for longer and in the same
        # way: the tags describe the STORE, so they are exactly as unconsumed under transpose as
        # forward, but only the forward entry asked. Measured before the fix, on the lattice fixture
        # in `test_component_harness_symmetry.jl`: a tagged collection was refused in a forward matvec
        # and ran in a transposed one, returning 1.43 relative error against the correctly reduced
        # map -- a full-magnitude wrong answer, not a perturbation.
        #
        # All six combinations now reach ONE guard site inside `applycoupling!` (#180), rather than
        # six hand-written calls that had to be kept in step. They are still all exercised, because
        # what matters is that every combination refuses, not how many places implement the refusal.
        pair = first(keys(collection.dict))
        buf = zeros(ComplexF64, 4)
        context = H2Factory.TranslationContext(SerialScheduler(), nothing)
        for direction in (
            H2Factory.ForwardTranslation(),
            H2Factory.TransposeTranslation(),
            H2Factory.AdjointTranslation(),
        )
            for accumulation in (H2Factory.Overwrite(), H2Factory.Accumulate())
                @test_throws "symmetry-reduced" H2Factory.applycoupling!(
                    buf, buf, collection, pair[1], pair[2], direction, accumulation, context
                )
            end
        end
    end

    # The refusal must not fire on collections that store one translation per direction.
    @test H2Factory.assertnosymmetry(plaincollection) === nothing
end

# A wrapper that holds a collection and forgets to forward the guard. Nothing about it is unusual --
# it is the shape `ToeplitzI2OTranslationCollection` and iFMM's `BsplineI2OTranslationCollection`
# both have -- which is the point: this is what the next wrapper will look like.
struct UnforwardingWrapper{C} <: H2Factory.AbstractI2OTranslationCollection
    inner::C
end
Base.getindex(w::UnforwardingWrapper, key) = w.inner[key]
Base.keys(w::UnforwardingWrapper) = keys(w.inner)
H2Factory.i2otrait(w::UnforwardingWrapper) = H2Factory.i2otrait(w.inner)

@testset "A wrapper must declare, not inherit" begin
    # THE NEGATIVE CONTROL FOR THE PROTOCOL ITSELF. While `assertnosymmetry` had a no-op fallback on
    # `AbstractI2OTranslationCollection`, this wrapper passed -- not because anyone decided it was
    # safe, but because everything passed. The tags inside were ignored and the orbit's canonical
    # translation applied to every pair.
    #
    # So the fallback is gone and the abstract-type method errors. A wrapper now has to say which it
    # is. This test fails if a permissive fallback is ever reintroduced, which is the only way the
    # hole can come back.
    rng = Xoshiro(7)
    points = [SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) for _ in 1:2000]
    tree = H2Trees.TwoNTree(
        points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.3, minvalues=0)
    )
    plans = H2Trees.buildplans(tree)
    plan = H2Trees.translatingplan(plans.trialaggregationplan, plans.testdisaggregationplan)

    tagged, _ = H2Factory.I2OTranslationCollection(
        tree,
        plan,
        H2Trees.SymmetryDirectionInvariancePerLevel(H2Trees.FullLatticeSymmetry()),
        DirectionRecordingTranslator(),
    )
    plain, _ = H2Factory.I2OTranslationCollection(
        tree, plan, H2Trees.DirectionInvariancePerLevel(), DirectionRecordingTranslator()
    )

    pair = first(keys(tagged.dict))
    buf = zeros(ComplexF64, 4)

    # Refused for NOT DECLARING, whatever it wraps. A wrapper around an untagged collection is
    # refused too: the protocol is about the declaration, not about whether this particular instance
    # happens to be safe. Anything else and the check would pass on today's data and fail on
    # tomorrow's, which is the failure mode it replaces.
    context = H2Factory.TranslationContext(SerialScheduler(), nothing)
    for inner in (tagged, plain)
        wrapper = UnforwardingWrapper(inner)
        @test_throws "assertnosymmetry is not declared" H2Factory.assertnosymmetry(wrapper)
        for direction in (
            H2Factory.ForwardTranslation(),
            H2Factory.TransposeTranslation(),
            H2Factory.AdjointTranslation(),
        )
            for accumulation in (H2Factory.Overwrite(), H2Factory.Accumulate())
                @test_throws "assertnosymmetry is not declared" H2Factory.applycoupling!(
                    buf, buf, wrapper, pair[1], pair[2], direction, accumulation, context
                )
            end
        end
    end

    # And the permissions that do exist are each the type's OWN, not the abstract method reached by
    # inheritance. `SymmetryReducedI2OTranslationCollection` is the one this catches: its docstring
    # has always said it must pass `assertnosymmetry`, but until the method was written the claim
    # was served by the fallback, and dispatch said so -- `which` resolved to the abstract method.
    #
    # ASKING `which` RATHER THAN CALLING. Calling and getting `nothing` passes either way, so it
    # cannot tell a declaration from an inheritance, which is the entire distinction here.
    reduced = H2Factory.symmetryreducedcollection(
        tagged, (receiving, symmetry) -> [1, 2, 3]
    )
    abstractmethod = which(
        H2Factory.assertnosymmetry, Tuple{UnforwardingWrapper{typeof(plain)}}
    )
    for collection in
        (plain, reduced, H2Factory.ToeplitzI2OTranslationCollection(plain, tree))
        @test which(H2Factory.assertnosymmetry, Tuple{typeof(collection)}) !==
            abstractmethod
        @test H2Factory.assertnosymmetry(collection) === nothing
    end
end

end # module TestSymmetryI2OCollection
