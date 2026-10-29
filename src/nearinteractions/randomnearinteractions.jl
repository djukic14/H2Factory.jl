"""
    RandomNearInteractionsAssembler{T}(operator, testspace, trialspace, quadstrategy)

Build a sparse near-interaction matrix with the normal sparsity pattern and random stored values.
The assembly method is provided when the required finite-element extension is loaded.
"""
struct RandomNearInteractionsAssembler{
    T,OperatorType,TestSpaceType,TrialSpaceType,QuadStratType
} <: AbstractNearInteractionsAssembler{T}
    operator::OperatorType
    testspace::TestSpaceType
    trialspace::TrialSpaceType
    quadstrat::QuadStratType

    function RandomNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end
