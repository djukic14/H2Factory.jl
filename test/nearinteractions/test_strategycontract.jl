module TestInteractionStrategyContract

using Test
using H2Factory
using H2Trees

# THE SUBTYPE CONTRACT THE FOUR STRATEGY ACCESSORS PROMISE, MADE EXECUTABLE.
#
# The accessors let a strategy subtype choose how it stores its data.
#
# No package in this repository or downstream currently does that. The promise was therefore made in
# documentation and nowhere else, and an unused extension point is exactly the kind that rots: the
# defaults would keep passing every existing test if someone replaced the accessors with direct
# `strategy.isnear` field reads, and the contract would be silently gone.
#
# These declarations live in `H2Factory.Interfaces`, which is a statement that they are protocol.
# This file is what makes that statement checkable rather than ceremonial.

# Deliberately DIFFERENT field names from `InteractionStrategy`'s, so the defaults cannot satisfy it:
# a direct field read would `ErrorException` here rather than return something plausible.
struct RenamedFieldStrategy{I,T,A,M} <: H2Factory.AbstractInteractionStrategy
    near::I
    iterator::T
    aggregate::A
    mode::M
end

H2Factory.nearcriterion(s::RenamedFieldStrategy) = s.near
H2Factory.translatingnodesiterator(s::RenamedFieldStrategy) = s.iterator
H2Factory.aggregatenode(s::RenamedFieldStrategy) = s.aggregate
H2Factory.aggregationmode(s::RenamedFieldStrategy) = s.mode

@testset "a strategy may store its data under its own field names" begin
    strategy = RenamedFieldStrategy(
        :nearmarker, :iteratormarker, :aggregatemarker, :modemarker
    )

    # The four accessors resolve through the subtype's own methods.
    @test H2Factory.nearcriterion(strategy) === :nearmarker
    @test H2Factory.translatingnodesiterator(strategy) === :iteratormarker
    @test H2Factory.aggregatenode(strategy) === :aggregatemarker
    @test H2Factory.aggregationmode(strategy) === :modemarker

    # NON-VACUITY, and the reason this type renames its fields. Were the accessors replaced by
    # direct field reads, the four assertions above would not merely fail, they could not run: this
    # type has no `isnear` field at all. That is what makes the test prove the indirection exists
    # rather than prove four values round-trip.
    @test !hasfield(RenamedFieldStrategy, :isnear)
    @test !hasfield(RenamedFieldStrategy, :translatingnodesiterator)
    @test !hasfield(RenamedFieldStrategy, :aggregatenode)
    @test !hasfield(RenamedFieldStrategy, :aggregationmode)

    # The overrides are methods on H2Factory's OWN generics, not same-named rivals defined here.
    # Without this the test would pass just as happily against a local function, which is the exact
    # failure the `import`-not-`using` discipline in `H2Factory.jl` exists to prevent.
    for (f, n) in (
        (H2Factory.nearcriterion, :nearcriterion),
        (H2Factory.translatingnodesiterator, :translatingnodesiterator),
        (H2Factory.aggregatenode, :aggregatenode),
        (H2Factory.aggregationmode, :aggregationmode),
    )
        @test f === getfield(H2Factory.Interfaces, n)
        @test which(f, Tuple{RenamedFieldStrategy}).module === @__MODULE__
    end
end

@testset "the defaults read the standard fields" begin
    # The subtype contract must not have cost the convenience default that every
    # `InteractionStrategy`-shaped type relies on.
    default = H2Factory.InteractionStrategy(;
        isnear=:d1, translatingnodesiterator=:d2, aggregatenode=:d3, aggregationmode=:d4
    )
    @test H2Factory.nearcriterion(default) === :d1
    @test H2Factory.translatingnodesiterator(default) === :d2
    @test H2Factory.aggregatenode(default) === :d3
    @test H2Factory.aggregationmode(default) === :d4

    # And it is the ABSTRACT-type default doing that work, not a concrete method.
    @test which(H2Factory.nearcriterion, Tuple{typeof(default)}).sig ==
        Tuple{typeof(H2Factory.nearcriterion),H2Factory.AbstractInteractionStrategy}
end

@testset "the strategy accessor does not reuse H2Trees.isnear" begin
    @test :nearcriterion in names(H2Factory)
    @test :isnear ∉ names(H2Factory)
    @test H2Factory.nearcriterion !== H2Trees.isnear
end

end # module TestInteractionStrategyContract
