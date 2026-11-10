module H2BlockSparse
using H2Factory
using BlockSparseMatrices
using OhMyThreads: SerialScheduler

function (storage::H2Factory.BlockSparseStorage)(
    blocks::AbstractVector{<:AbstractMatrix}, rows, columns, size::Tuple{Int,Int}
)
    # Coloring an empty block list fails, and there is no work to distribute.
    scheduler = isempty(blocks) ? SerialScheduler() : storage.scheduler
    return BlockSparseMatrix(blocks, rows, columns, size; scheduler=scheduler)
end

end # module H2BlockSparse
