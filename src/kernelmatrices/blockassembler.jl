"""
    AbstractStorageTrait

Base type for kernel-matrix storage traits. An [`AssemblyStrategy`](@ref) holds one and
[`assembleblocks`](@ref) dispatches on it to select the stored format.
"""
abstract type AbstractStorageTrait end

"""
    SparseStorage()

Store assembled blocks in one `SparseMatrixCSC`.
"""
struct SparseStorage <: AbstractStorageTrait end

"""
    BlockSparseStorage(; scheduler=SerialScheduler())

Store assembled blocks as a `BlockSparseMatrix`. Requires BlockSparseMatrices to be loaded.

Calling `storage(blocks, rows, columns, size)` returns the `BlockSparseMatrix` of the dense
`blocks` with global indices `rows[i]` and `columns[i]`. Its matrix-vector products use
`scheduler`, which is independent of the assembly scheduler.
"""
struct BlockSparseStorage{Scheduler} <: AbstractStorageTrait
    scheduler::Scheduler
end

BlockSparseStorage(; scheduler=SerialScheduler()) = BlockSparseStorage(scheduler)

# The BlockSparseMatrices extension adds the method for dense blocks and a `(rows, columns)`
# size, which returns the `BlockSparseMatrix` using `storage.scheduler`.
function (storage::BlockSparseStorage)(blocks, rows, columns, size)
    return error(
        "BlockSparseStorage needs the BlockSparseMatrices package. Load " *
        "BlockSparseMatrices to assemble into `BlockSparseStorage`.",
    )
end

"""
    assembleblocks(kernelmatrix, rows, columns, storage, scheduler; verbose=false)

Evaluate the blocks `kernelmatrix[rows[i], columns[i]]` and store them in the format
selected by `storage`.

`scheduler` distributes the blocks over tasks, so `kernelmatrix` must support concurrent
block evaluation. Storage traits provided by extensions add methods for their trait type.
"""
function assembleblocks(
    kernelmatrix::AbstractKernelMatrix,
    rows,
    columns,
    ::SparseStorage,
    scheduler;
    verbose=false,
)
    @assert length(rows) == length(columns)
    T = eltype(kernelmatrix)

    # Prefix sums give each block a separate slice of the triplet buffers.
    offsets = Vector{Int}(undef, length(rows) + 1)
    offsets[begin] = 0
    tforeach(eachindex(rows); scheduler=scheduler) do i
        offsets[i + 1] = length(rows[i]) * length(columns[i])
        return nothing
    end
    cumsum!(offsets, offsets)

    I = Vector{Int}(undef, offsets[end])
    J = Vector{Int}(undef, offsets[end])
    V = zeros(T, offsets[end])

    pbar = progressbar(length(rows) + 2, verbose; desc="Computing nearinteractions")

    tforeach(eachindex(rows); scheduler=scheduler) do i
        slice = (offsets[i] + 1):offsets[i + 1]
        blockrows = rows[i]
        blockcolumns = columns[i]
        nrows = length(blockrows)
        # Column-major order matches the reshaped value block below.
        for (n, column) in enumerate(blockcolumns), (m, row) in enumerate(blockrows)
            k = offsets[i] + (n - 1) * nrows + m
            I[k] = row
            J[k] = column
        end
        block = reshape(view(V, slice), nrows, length(blockcolumns))
        kernelmatrix(block, blockrows, blockcolumns)
        next!(pbar)
        return nothing
    end

    next!(pbar; desc="Computing sparse representation")
    matrix = SparseArrays.sparse(I, J, V, size(kernelmatrix)...)
    next!(pbar; desc="Finished computation of nearinteractions")
    finish!(pbar)

    return matrix
end

function assembleblocks(
    kernelmatrix::AbstractKernelMatrix,
    rows,
    columns,
    storage::BlockSparseStorage,
    scheduler;
    verbose=false,
)
    @assert length(rows) == length(columns)
    T = eltype(kernelmatrix)
    blocks = Vector{Matrix{T}}(undef, length(rows))

    pbar = progressbar(length(rows) + 2, verbose; desc="Computing nearinteractions")

    tforeach(eachindex(rows); scheduler=scheduler) do i
        block = zeros(T, length(rows[i]), length(columns[i]))
        kernelmatrix(block, rows[i], columns[i])
        blocks[i] = block
        next!(pbar)
        return nothing
    end

    next!(pbar; desc="Computing block-sparse representation")
    matrix = storage(blocks, rows, columns, size(kernelmatrix))
    next!(pbar; desc="Finished computation of nearinteractions")
    finish!(pbar)

    return matrix
end
