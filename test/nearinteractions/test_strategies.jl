module TestNearInteractionsStrategies

using Test
using H2Factory
using H2Trees

# The strategy layer is a thin selection/configuration layer over the assemblers. It needs no mesh,
# no quadrature and no BEAST: every assembler here stores its arguments generically, so sentinels
# are enough to pin the dispatch and the defaulting. Keeping this tier BEAST-free is deliberate --
# it is the part of the runtime that should be testable without the heavy mesh stack.

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

@testset "buildnearinteractionsassembler: strategy selects the assembler, γ lands in the type" begin
    # The selection layer's whole job: map a strategy to a concrete assembler, forwarding the
    # operator/spaces and putting γ (the map's scalar type) into the assembler's leading type
    # parameter -- which is what `AbstractNearInteractionsAssembler{T}` consumers dispatch on.
    γ = 1.0 + 2.0im
    for (strategy, Assembler) in (
        (
            H2Factory.DummyNearInteractionsStrategy(:quadstrategy),
            H2Factory.DummyNearInteractionsAssembler,
        ),
        (
            H2Factory.BlockBEASTNearInteractionsStrategy(:quadstrategy),
            H2Factory.BlockBEASTNearInteractionsAssembler,
        ),
    )
        @test strategy isa H2Factory.AbstractNearInteractionsStrategy

        assembler = H2Factory.buildnearinteractionsassembler(
            :operator, :testspace, :trialspace, γ, strategy
        )

        @test assembler isa Assembler
        @test assembler isa H2Factory.AbstractNearInteractionsAssembler{typeof(γ)}
        # arguments forwarded in the documented order
        @test assembler.operator === :operator
        @test assembler.testspace === :testspace
        @test assembler.trialspace === :trialspace
        # the strategy's quadstrategy is what reaches the assembler
        @test assembler.quadstrat === :quadstrategy
    end
end

# An `isnear` that accepts any "tree" -- assembly evaluates `nearcriterion(strategy)(tree)`.
# before handing off to the assembler, and the pass-through path must not need a real tree.
struct AnyIsNear end
(::AnyIsNear)(tree) = :nearpredicate

# Minimal H2 map: `nearmul!` only ever reaches the near block through `nearinteractions(A)`, so this
# is all that is needed to prove a passed-through matrix is what the matvec actually multiplies.
struct PassThroughFakeMap{T,N} <: H2Factory.GalerkinH2Map{T}
    nearinteractions::N
end
PassThroughFakeMap(near) = PassThroughFakeMap{eltype(near),typeof(near)}(near)

@testset "PassThroughNearInteractionsAssembler: reuses an already-computed near matrix" begin
    # The point of this assembler: near-field quadrature is often the dominant assembly cost, so a
    # near block computed once (shared between maps, reused across solves, loaded from disk) should
    # be usable as-is instead of reassembled. Nothing here may touch the operator/spaces/tree --
    # sentinels in those slots prove it.
    nearmatrix = ComplexF64[1 2; 3 4]
    γ = 1.0 + 0.0im

    assembler = H2Factory.PassThroughNearInteractionsAssembler{typeof(γ)}(
        :operator, :testspace, :trialspace, nearmatrix
    )

    # Consistent with the real assemblers: γ is the LEADING type parameter, so this is a drop-in
    # wherever an `AbstractNearInteractionsAssembler{T}` is expected. The stored matrix's type is a
    # separate parameter rather than displacing γ.
    @test assembler isa H2Factory.AbstractNearInteractionsAssembler{typeof(γ)}
    @test H2Factory.nearinteractions(assembler) === nearmatrix

    # Returns the very same object -- not a copy, and without walking any tree.
    @test assembler(:tree, :isnear) === nearmatrix
    @test assembler(nothing, nothing; verbose=true) === nearmatrix
end

@testset "PassThroughNearInteractionsStrategy: precomputed matrix reaches the H2 map's matvec" begin
    # End-to-end on the real selection path: strategy -> buildnearinteractionsassembler ->
    # assemblenearinteractions -> H2 map -> nearmul!. Sentinel operator/spaces/tree throughout: if
    # any stage tried to actually assemble, it would fail rather than silently reassemble.
    nearmatrix = ComplexF64[1 2; 3 4]
    γ = 1.0 + 0.0im
    strategy = H2Factory.PassThroughNearInteractionsStrategy(nearmatrix)

    @test strategy isa H2Factory.AbstractNearInteractionsStrategy

    assembled = H2Factory.assemblenearinteractions(
        :operator,
        :testspace,
        :trialspace,
        :tree,
        γ,
        strategy,
        H2Factory.InteractionStrategy(; isnear=AnyIsNear());
        verbose=false,
    )
    # Identity, not just equality: the matrix must arrive untouched and uncopied.
    @test assembled === nearmatrix

    # ...and the map really multiplies by it.
    A = PassThroughFakeMap(assembled)
    @test H2Factory.nearinteractions(A) === nearmatrix

    x = ComplexF64[1.0, -2.0]
    @test H2Factory.nearmul(A, x) ≈ nearmatrix * x
    @test H2Factory.nearmul(transpose(A), x) ≈ transpose(nearmatrix) * x
    @test H2Factory.nearmul(adjoint(A), x) ≈ adjoint(nearmatrix) * x

    y = similar(x)
    H2Factory.nearmul!(y, A, x)
    @test y ≈ nearmatrix * x
    # α/β form: y = 2*(N*x) + 3*y
    y0 = ComplexF64[5.0, 7.0]
    y = copy(y0)
    H2Factory.nearmul!(y, A, x, 2, 3)
    @test y ≈ 2 * (nearmatrix * x) + 3 * y0
end

end # module TestNearInteractionsStrategies
