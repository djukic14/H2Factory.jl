"""
    SparseBEASTNearInteractionsAssembler{T}(operator, testspace, trialspace, quadstrategy)

Assemble near interactions into a sparse matrix of scalar type `T`. The assembly method is
available when the required finite-element extension is loaded.
"""
struct SparseBEASTNearInteractionsAssembler{
    T,OperatorType,TestSpaceType,TrialSpaceType,QuadStratType
} <: AbstractNearInteractionsAssembler{T}
    operator::OperatorType
    testspace::TestSpaceType
    trialspace::TrialSpaceType
    quadstrat::QuadStratType

    function SparseBEASTNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end
