module H2BlockSparse
using BEAST
using H2Factory
import H2Factory: progressbar
using ProgressMeter
using BlockSparseMatrices

function assembleblockprimer(assembler)
    return BEAST.assembleblock_primer(
        H2Factory.operator(assembler),
        H2Factory.testspace(assembler),
        H2Factory.trialspace(assembler);
        quadstrat=H2Factory.quadstrategy(assembler),
    )
end

include("assembly.jl")
include("blocksparseassembler.jl")

end # module H2BlockSparse
