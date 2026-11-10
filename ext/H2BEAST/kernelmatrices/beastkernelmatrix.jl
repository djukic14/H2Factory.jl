function H2Factory.BEASTKernelMatrix{T}(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
) where {T}
    # BEAST wraps `quadstrat` itself when one geometry refines the other, and builds the
    # primer from the wrapped strategy.
    blockassembler = BEAST.blockassembler(operator, testspace, trialspace; quadstrat)
    return H2Factory.BEASTKernelMatrix{T}(
        blockassembler, numfunctions(testspace), numfunctions(trialspace)
    )
end

function H2Factory.BEASTKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    T = BEAST.scalartype(operator, testspace, trialspace)
    return H2Factory.BEASTKernelMatrix{T}(operator, testspace, trialspace, quadstrat)
end

function H2Factory.defaultmatrixdata(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.defaultquadstrat(operator, testspace, trialspace)
end

function H2Factory.buildkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    quadstrat,
)
    return H2Factory.BEASTKernelMatrix(operator, testspace, trialspace, quadstrat)
end

struct BlockStoreFunctor{M}
    block::M
end

function (f::BlockStoreFunctor)(v, m, n)
    # BEAST passes indices local to the requested block.
    f.block[m, n] += v
    return nothing
end

# BEAST expects index collections; a single index becomes a one-element range.
_blockindices(index::Integer) = index:index
_blockindices(indices) = indices

function (matrix::H2Factory.BEASTKernelMatrix)(block, rows, columns)
    matrix.blockassembler(
        _blockindices(rows), _blockindices(columns), BlockStoreFunctor(block)
    )
    return nothing
end

function H2Factory.buildkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    ::H2Factory.DummyEntries,
)
    T = BEAST.scalartype(operator, testspace, trialspace)
    return H2Factory.DummyKernelMatrix{T}(numfunctions(testspace), numfunctions(trialspace))
end

function H2Factory.buildkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    ::H2Factory.RandomEntries,
)
    T = BEAST.scalartype(operator, testspace, trialspace)
    nrows, ncolumns = numfunctions(testspace), numfunctions(trialspace)
    return H2Factory.RandomKernelMatrix{T}(nrows, ncolumns)
end
