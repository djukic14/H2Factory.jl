"""
    AbstractNearInteractionsStrategy

Base type for strategies that select a near-interaction assembler. New strategies implement
[`buildnearinteractionsassembler`](@ref).
"""
abstract type AbstractNearInteractionsStrategy end

"""
    BlockBEASTNearInteractionsStrategy(quadstrategy)

Selects [`BlockBEASTNearInteractionsAssembler`](@ref) as the near-interactions assembler.
"""
struct BlockBEASTNearInteractionsStrategy{Q} <: AbstractNearInteractionsStrategy
    quadstrategy::Q
end

"""
    DummyNearInteractionsStrategy(quadstrategy)

Select [`DummyNearInteractionsAssembler`](@ref), which returns an all-zero near block.
"""
struct DummyNearInteractionsStrategy{Q} <: AbstractNearInteractionsStrategy
    quadstrategy::Q
end

"""
    PassThroughNearInteractionsStrategy(nearinteractions)

Select [`PassThroughNearInteractionsAssembler`](@ref), which returns `nearinteractions` unchanged.
The matrix must already match the map's tree and spaces.
"""
struct PassThroughNearInteractionsStrategy{N} <: AbstractNearInteractionsStrategy
    nearinteractions::N
end

function buildnearinteractionsassembler(
    operator, testspace, trialspace, γ, strategy::PassThroughNearInteractionsStrategy
)
    return PassThroughNearInteractionsAssembler{typeof(γ)}(
        operator, testspace, trialspace, strategy.nearinteractions
    )
end

function buildnearinteractionsassembler(
    operator, testspace, trialspace, γ, strategy::BlockBEASTNearInteractionsStrategy
)
    return BlockBEASTNearInteractionsAssembler{typeof(γ)}(
        operator, testspace, trialspace, strategy.quadstrategy
    )
end

function buildnearinteractionsassembler(
    operator, testspace, trialspace, γ, strategy::DummyNearInteractionsStrategy
)
    return DummyNearInteractionsAssembler{typeof(γ)}(
        operator, testspace, trialspace, strategy.quadstrategy
    )
end

"""
    assemblenearinteractions(operator, testspace, trialspace, tree, γ, nearinteractionsstrategy, interactionstrategy; verbose=false)

Build and run the near-interaction assembler selected by `nearinteractionsstrategy`, using the near
predicate from `interactionstrategy`.
"""
function assemblenearinteractions(
    operator,
    testspace,
    trialspace,
    tree,
    γ,
    nearinteractionsstrategy::AbstractNearInteractionsStrategy,
    interactionstrategy::AbstractInteractionStrategy;
    verbose=false,
)
    nearassembler = buildnearinteractionsassembler(
        operator, testspace, trialspace, γ, nearinteractionsstrategy
    )
    return nearassembler(tree, nearcriterion(interactionstrategy)(tree); verbose=verbose)
end
