"""
    BEASTKernelMatrix{T,A} <: AbstractKernelMatrix{T}

Kernel matrix of a BEAST integral operator between a test and a trial space. Requires BEAST
to be loaded.

    BEASTKernelMatrix{T}(operator, testspace, trialspace[, quadstrat])
    BEASTKernelMatrix(operator, testspace, trialspace[, quadstrat])

`quadstrat` defaults to `BEAST.defaultquadstrat(operator, testspace, trialspace)`. Without
`T`, the scalar type is `BEAST.scalartype(operator, testspace, trialspace)`. An explicit `T`
only sets the type entries are stored in; BEAST evaluates them in its own scalar type.

The BEAST block assembler `blockassembler::A` is built once on construction and shared by
all blocks. It is safe to evaluate blocks from concurrent tasks.
"""
struct BEASTKernelMatrix{T,A} <: AbstractKernelMatrix{T}
    blockassembler::A
    nrows::Int
    ncolumns::Int

    function BEASTKernelMatrix{T}(
        blockassembler, nrows::Integer, ncolumns::Integer
    ) where {T}
        return new{T,typeof(blockassembler)}(blockassembler, nrows, ncolumns)
    end
end

Base.size(matrix::BEASTKernelMatrix) = (matrix.nrows, matrix.ncolumns)
