function H2Factory.assemble(
    assembler::Union{
        <:H2Factory.SparseBEASTNearInteractionsAssembler,
        <:H2Factory.RandomNearInteractionsAssembler,
    },
    values,
    nearvalues,
    verbose::Bool,
)
    return _sparseassemble(assembler, values, nearvalues, verbose)
end

struct SparseStoreFunctor{VecI,VecJ,VecV,T,S}
    I::VecI
    J::VecJ
    V::VecV
    tdata::T
    sdata::S
end

function (f::SparseStoreFunctor)(v, m, n)
    # Map block-local indices to global basis-function IDs.
    f.I[(n - 1) * length(f.tdata) + m] = f.tdata[m]
    f.J[(n - 1) * length(f.tdata) + m] = f.sdata[n]
    f.V[(n - 1) * length(f.tdata) + m] += v
    return nothing
end

function assembleblock(
    ::SparseBEASTNearInteractionsAssembler, I, J, V, tdata, sdata, blkasm
)
    blkasm(tdata, sdata, SparseStoreFunctor(I, J, V, tdata, sdata))
    return nothing
end

function assembleblockprimer(assembler)
    return BEAST.assembleblock_primer(
        H2Factory.operator(assembler),
        H2Factory.testspace(assembler),
        H2Factory.trialspace(assembler);
        quadstrat=H2Factory.quadstrategy(assembler),
    )
end

function blockassembler(
    assembler::SparseBEASTNearInteractionsAssembler; primer=assembleblockprimer(assembler)
)
    operator = H2Factory.operator(assembler)
    testspace = H2Factory.testspace(assembler)
    trialspace = H2Factory.trialspace(assembler)
    quadstrat = H2Factory.quadstrategy(assembler)
    # Reuse the primer only when BEAST's method accepts it.
    signature = Tuple{typeof(operator),typeof(testspace),typeof(trialspace)}
    if hasmethod(BEAST.blockassembler, signature, (:quadstrat, :primer))
        return BEAST.blockassembler(operator, testspace, trialspace; quadstrat, primer)
    end
    return BEAST.blockassembler(operator, testspace, trialspace; quadstrat)
end

function Base.size(
    assembler::Union{
        <:H2Factory.RandomNearInteractionsAssembler,
        <:H2Factory.SparseBEASTNearInteractionsAssembler,
        <:H2Factory.BlockBEASTNearInteractionsAssembler,
    },
)
    return (numfunctions(assembler.testspace), numfunctions(assembler.trialspace))
end
