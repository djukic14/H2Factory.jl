"""
    DummyI2OTranslationCollection()

A coupling collection that leaves the output unchanged in every direction, including
`Overwrite()` mode. It stores no operators.
"""
struct DummyI2OTranslationCollection <: AbstractI2OCouplingCollection end

function DummyI2OTranslationCollection(
    tree, aggregationplan::A, disaggregationplan::D, translationtrait, Translator; kwargs...
) where {A<:H2Trees.AbstractAggregationPlan,D<:H2Trees.AbstractDisaggregationPlan}
    return DummyI2OTranslationCollection(
        tree,
        H2Trees.translatingplan(aggregationplan, disaggregationplan),
        translationtrait,
        Translator;
        kwargs...,
    )
end

function DummyI2OTranslationCollection(
    tree, translatingplan, translationtrait, Translator; kwargs...
)
    return DummyI2OTranslationCollection()
end

assertnosymmetry(::DummyI2OTranslationCollection) = nothing

i2otrait(::DummyI2OTranslationCollection) = DummyTranslation()

# No coupling is applied, in any direction or accumulation mode.
function applycouplingkernel!(
    out,
    ::Any,
    ::DummyI2OTranslationCollection,
    ::DummyTranslation,
    ::Int,
    ::Int,
    ::AbstractTranslationDirection,
    ::AbstractAccumulationMode,
    ::TranslationContext,
)
    return out
end
