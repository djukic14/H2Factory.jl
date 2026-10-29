"""
    BlockBEASTNearInteractionsAssembler{T}(operator, testspace, trialspace, quadstrategy)

Assemble near interactions into a block-sparse matrix of scalar type `T`. The assembly method is
available when the required block-sparse extension is loaded.
"""
struct BlockBEASTNearInteractionsAssembler{
    T,OperatorType,TestSpaceType,TrialSpaceType,QuadStratType
} <: AbstractNearInteractionsAssembler{T}
    operator::OperatorType
    testspace::TestSpaceType
    trialspace::TrialSpaceType
    quadstrat::QuadStratType

    function BlockBEASTNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end

function supportsgalerkinsymmetric(::Type{<:BlockBEASTNearInteractionsAssembler})
    return false
end
