"""
    DummyNearInteractionsAssembler{T}(operator, testspace, trialspace, quadstrategy)

Build an all-zero sparse near-interaction matrix of scalar type `T`. This is useful when only the
far-field part of a map should be active. Calling the assembler with a tree derives the matrix size
from that tree and does not require a finite-element extension.
"""
struct DummyNearInteractionsAssembler{
    T,OperatorType,TestSpaceType,TrialSpaceType,QuadStratType
} <: AbstractNearInteractionsAssembler{T}
    operator::OperatorType
    testspace::TestSpaceType
    trialspace::TrialSpaceType
    quadstrat::QuadStratType

    function DummyNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end

function (::DummyNearInteractionsAssembler{T})(
    tree, ::H2Trees.isBlockTree, _isnear; kwargs...
) where {T}
    nrows = H2Trees.numberofvalues(H2Trees.testtree(tree))
    ncols = H2Trees.numberofvalues(H2Trees.trialtree(tree))
    return spzeros(T, nrows, ncols)
end

function (::DummyNearInteractionsAssembler{T})(
    tree, ::H2Trees.AbstractTreeTrait, _isnear; kwargs...
) where {T}
    n = H2Trees.numberofvalues(tree)
    return spzeros(T, n, n)
end
