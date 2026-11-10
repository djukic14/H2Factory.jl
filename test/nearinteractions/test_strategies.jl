module TestNearInteractionsStrategies

using Test
using H2Factory
using H2Trees

# The interaction strategy is a thin configuration layer. It needs no mesh, no quadrature
# and no BEAST: sentinels are enough to pin the defaulting. Keeping this tier BEAST-free is
# deliberate -- it is the part of the runtime that should be testable without the heavy mesh
# stack.

# Sentinels: distinct singleton types, so a wrong default shows up as a type mismatch rather than a
# silently-equal value.
struct SentinelIsNear end
struct SentinelMode end

@testset "InteractionStrategy: defaults derive from `isnear` and stay consistent" begin
    # THE contract this struct exists for. `translatingnodesiterator` and `aggregatenode` are
    # chained defaults off `isnear`, so a caller overriding `isnear` ALONE must not be left with the
    # other two still built from the default. Before the strategy structs those were three flat
    # kwargs and this desync was silent and easy to hit.
    isnear = SentinelIsNear()
    s = H2Factory.InteractionStrategy(; isnear=isnear)

    @test H2Factory.nearcriterion(s) === isnear

    # Compare against the objects an explicit, hand-chained construction produces. Type equality is
    # the discriminating check: had the default ignored our `isnear`, the chain would carry
    # H2Trees' default `isnear` type instead and these would differ.
    expectediterator = H2Trees.TranslatingNodesIterator(; isnear=isnear)
    expectedaggregate = H2Trees.istranslatingnode(;
        TranslatingNodesIterator=expectediterator
    )
    @test typeof(H2Factory.translatingnodesiterator(s)) === typeof(expectediterator)
    @test typeof(H2Factory.aggregatenode(s)) === typeof(expectedaggregate)

    # ...and `aggregatenode` must be built from THIS strategy's iterator instance, not merely one of
    # the same type -- identity, not just equality.
    @test H2Factory.aggregatenode(s).TranslatingNodesIterator ===
        H2Factory.translatingnodesiterator(s)

    # A default `isnear` must NOT leak the sentinel: guards against the test passing vacuously
    # because everything happens to be the same type.
    d = H2Factory.InteractionStrategy()
    @test typeof(H2Factory.translatingnodesiterator(d)) !== typeof(expectediterator)
end

@testset "InteractionStrategy: explicit arguments win over the derived defaults" begin
    # Chained defaults must not override what the caller actually passed.
    isnear = SentinelIsNear()
    iterator = H2Trees.TranslatingNodesIterator(; isnear=isnear)
    aggregate = H2Trees.istranslatingnode(; TranslatingNodesIterator=iterator)
    mode = SentinelMode()

    s = H2Factory.InteractionStrategy(;
        isnear=isnear,
        translatingnodesiterator=iterator,
        aggregatenode=aggregate,
        aggregationmode=mode,
    )

    @test H2Factory.nearcriterion(s) === isnear
    @test H2Factory.translatingnodesiterator(s) === iterator
    @test H2Factory.aggregatenode(s) === aggregate
    @test H2Factory.aggregationmode(s) === mode
    @test s isa H2Factory.AbstractInteractionStrategy
end

@testset "InteractionStrategy: default aggregationmode" begin
    @test H2Factory.aggregationmode(H2Factory.InteractionStrategy()) ==
        H2Trees.AggregateMode()
end

end # module TestNearInteractionsStrategies
