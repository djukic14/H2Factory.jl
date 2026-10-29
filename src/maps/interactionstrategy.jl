"""
    AbstractInteractionStrategy

Base type for strategies that divide interactions into near and far work and configure the tree
traversal.
"""
abstract type AbstractInteractionStrategy end

"""
    InteractionStrategy(; isnear, translatingnodesiterator, aggregatenode, aggregationmode)

Hold the interaction-partition and traversal settings used to build an H2 map.
`translatingnodesiterator` and `aggregatenode` are derived from `isnear` by default.
"""
struct InteractionStrategy{IsNear,TNI,AN,AM} <: AbstractInteractionStrategy
    isnear::IsNear
    translatingnodesiterator::TNI
    aggregatenode::AN
    aggregationmode::AM
end

function InteractionStrategy(;
    isnear=H2Trees.isnear(),
    translatingnodesiterator=H2Trees.TranslatingNodesIterator(; isnear=isnear),
    aggregatenode=H2Trees.istranslatingnode(;
        TranslatingNodesIterator=translatingnodesiterator
    ),
    aggregationmode=H2Trees.AggregateMode(),
)
    return InteractionStrategy(
        isnear, translatingnodesiterator, aggregatenode, aggregationmode
    )
end

# Default accessors read fields with the same names. Other layouts can override them.
nearcriterion(strategy::AbstractInteractionStrategy) = strategy.isnear

translatingnodesiterator(strategy::AbstractInteractionStrategy) =
    strategy.translatingnodesiterator

aggregatenode(strategy::AbstractInteractionStrategy) = strategy.aggregatenode

aggregationmode(strategy::AbstractInteractionStrategy) = strategy.aggregationmode

"""
    assembleplans(tree, strategy::AbstractInteractionStrategy)

Build H2Trees traversal plans from the four settings in `strategy`.
"""
function assembleplans(tree, strategy::AbstractInteractionStrategy)
    return H2Trees.buildplans(
        tree;
        builder=H2Trees.PlanBuilder(;
            isnear=nearcriterion(strategy),
            translatingnodesiterator=translatingnodesiterator(strategy),
            aggregatenode=aggregatenode(strategy),
            aggregationmode=aggregationmode(strategy),
        ),
    )
end
