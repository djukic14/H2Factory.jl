function H2Factory.assemble(
    assembler::H2Factory.BlockBEASTNearInteractionsAssembler,
    values,
    nearvalues,
    verbose::Bool,
)
    return _blocksparseassemble(assembler, values, nearvalues, verbose)
end

function H2Factory.assemble(
    assembler::H2Factory.BlockBEASTNearInteractionsAssembler,
    selfvalues,
    values,
    nearvalues,
    verbose::Bool,
)
    return _blocksparseassemble(assembler, selfvalues, values, nearvalues, verbose)
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    f.matrix[m, n] += v
    return nothing
end

function assembleblock(
    ::BlockBEASTNearInteractionsAssembler, matrixblock, tdata, sdata, blkasm
)
    blkasm(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end

function blockassembler(
    assembler::BlockBEASTNearInteractionsAssembler; primer=assembleblockprimer(assembler)
)
    operator = H2Factory.operator(assembler)
    testspace = H2Factory.testspace(assembler)
    trialspace = H2Factory.trialspace(assembler)
    quadstrat = H2Factory.quadstrategy(assembler)
    # Reuse the primer only when BEAST's public method accepts it.
    signature = Tuple{typeof(operator),typeof(testspace),typeof(trialspace)}
    if hasmethod(BEAST.blockassembler, signature, (:quadstrat, :primer))
        return BEAST.blockassembler(operator, testspace, trialspace; quadstrat, primer)
    end
    return BEAST.blockassembler(operator, testspace, trialspace; quadstrat)
end
