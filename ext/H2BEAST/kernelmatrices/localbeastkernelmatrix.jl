"""
    LocalBEASTKernelMatrix{T} <: AbstractKernelMatrix{T}

[`BEASTKernelMatrix`](@ref) prepared for sampling single rows against fixed `columns` and
single columns against fixed `rows`. Built by [`localkernelmatrix`](@ref) for
`BEAST.DoubleNumQStrat`, the regular quadrature of well-separated blocks.

Row data reduces the trial side to the elements of `columns` and column data the test side
to the elements of `rows`, so one row or column integrates only over these elements. Any
other request is passed to the wrapped matrix. Either data is `nothing` if that side is
never sampled. The data holds mutable scratch: one local matrix serves one task.
"""
struct LocalBEASTKernelMatrix{T,M,R,C} <: H2Factory.AbstractKernelMatrix{T}
    matrix::M
    rowdata::R
    columndata::C

    function LocalBEASTKernelMatrix(
        matrix::H2Factory.BEASTKernelMatrix{T}, rows, columns
    ) where {T}
        assembler = matrix.blockassembler
        rowdata = _localrowdata(assembler, columns)
        columndata = _localcolumndata(assembler, rows)
        return new{T,typeof(matrix),typeof(rowdata),typeof(columndata)}(
            matrix, rowdata, columndata
        )
    end
end

Base.size(matrix::LocalBEASTKernelMatrix) = size(matrix.matrix)

function H2Factory.localkernelmatrix(matrix::H2Factory.BEASTKernelMatrix, rows, columns)
    return _localkernelmatrix(matrix, rows, columns, matrix.blockassembler.quadstrat)
end

# Row and column sampling reuse the regular quadrature data of the whole space; singular
# strategies keep the block path.
_localkernelmatrix(matrix, rows, columns, quadstrat) = matrix

function _localkernelmatrix(matrix, rows, columns, ::BEAST.DoubleNumQStrat)
    return LocalBEASTKernelMatrix(matrix, rows, columns)
end

# The sorted elements supporting the functions `ids` of `space`, as BEAST's block
# assembler computes them.
function _activeelements(space, ids)
    cells = collect(Int, shape.cellid for id in ids for shape in space.fns[id])
    return unique!(sort!(cells))
end

# BEAST integrates in its own scalar type; an explicit `T` only sets the stored type.
function _localzlocal(assembler)
    T = BEAST.scalartype(assembler.biop, assembler.tfs, assembler.bfs)
    ntestshapes = size(assembler.testassemblydata.data, 2)
    ntrialshapes = size(assembler.trialassemblydata.data, 2)
    return zeros(T, ntestshapes, ntrialshapes)
end

_localrowdata(assembler, ::Nothing) = nothing

function _localrowdata(assembler, columns)
    testspace = assembler.tfs
    trialspace = assembler.bfs
    trialelements = _activeelements(trialspace, columns)
    return (;
        # A one-function test space whose function is replaced by the sampled row.
        testspace=BEAST.subset(testspace, 1:1),
        testshapes=BEAST.refspace(testspace),
        trialspace,
        trialelements=view(assembler.trialelements, trialelements),
        trialassemblydata=BEAST.reduce_assembly_data(
            assembler.trialassemblydata, columns, trialelements
        ),
        trialshapes=BEAST.refspace(trialspace),
        quadraturedata=(
            assembler.quadraturedata[1], view(assembler.quadraturedata[2], :, trialelements)
        ),
        zlocal=_localzlocal(assembler),
    )
end

_localcolumndata(assembler, ::Nothing) = nothing

function _localcolumndata(assembler, rows)
    testspace = assembler.tfs
    trialspace = assembler.bfs
    testelements = _activeelements(testspace, rows)
    return (;
        testspace,
        testelements=view(assembler.testelements, testelements),
        testassemblydata=BEAST.reduce_assembly_data(
            assembler.testassemblydata, rows, testelements
        ),
        testshapes=BEAST.refspace(testspace),
        # A one-function trial space whose function is replaced by the sampled column.
        trialspace=BEAST.subset(trialspace, 1:1),
        trialshapes=BEAST.refspace(trialspace),
        quadraturedata=(
            view(assembler.quadraturedata[1], :, testelements), assembler.quadraturedata[2]
        ),
        zlocal=_localzlocal(assembler),
    )
end

# One row against the prepared `columns`.
function (matrix::LocalBEASTKernelMatrix)(
    block, row::Integer, columns::AbstractVector{<:Integer}
)
    data = matrix.rowdata
    data === nothing && return matrix.matrix(block, row, columns)
    assembler = matrix.matrix.blockassembler
    testspace = data.testspace
    testspace.fns[1] = assembler.tfs.fns[row]
    testspace.pos[1] = assembler.tfs.pos[row]
    BEAST.assemblerow_body!(
        assembler.biop,
        testspace,
        assembler.testelements,
        data.testshapes,
        data.trialassemblydata,
        data.trialspace,
        data.trialelements,
        data.trialshapes,
        data.zlocal,
        data.quadraturedata,
        BlockStoreFunctor(block);
        quadstrat=assembler.quadstrat,
    )
    return nothing
end

# One column against the prepared `rows`.
function (matrix::LocalBEASTKernelMatrix)(
    block, rows::AbstractVector{<:Integer}, column::Integer
)
    data = matrix.columndata
    data === nothing && return matrix.matrix(block, rows, column)
    assembler = matrix.matrix.blockassembler
    trialspace = data.trialspace
    trialspace.fns[1] = assembler.bfs.fns[column]
    trialspace.pos[1] = assembler.bfs.pos[column]
    BEAST.assemblecol_body!(
        assembler.biop,
        data.testassemblydata,
        data.testspace,
        data.testelements,
        data.testshapes,
        trialspace,
        assembler.trialelements,
        data.trialshapes,
        data.zlocal,
        data.quadraturedata,
        BlockStoreFunctor(block);
        quadstrat=assembler.quadstrat,
    )
    return nothing
end

function (matrix::LocalBEASTKernelMatrix)(block, row::Integer, column::Integer)
    return matrix.matrix(block, row, column)
end

(matrix::LocalBEASTKernelMatrix)(block, rows, columns) = matrix.matrix(block, rows, columns)
