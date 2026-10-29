function _blocksparseassemble(
    assembler::H2Factory.AbstractNearInteractionsAssembler{T},
    values,
    nearvalues,
    verbose::Bool;
) where {T}
    blocks = Vector{Matrix{T}}(undef, length(values))
    addblocks!(blocks, values, nearvalues)
    @assert length(blocks) == length(values) == length(nearvalues)

    pbar = progressbar(length(values) + 1, verbose; desc="Computing nearinteractions")
    primer = assembleblockprimer(assembler)

    Threads.@threads for i in eachindex(values)
        assembleblock(
            assembler,
            blocks[i],
            values[i],
            nearvalues[i],
            blockassembler(assembler; primer=primer),
        )
        next!(pbar)
    end

    next!(pbar; desc="Finished computation of nearinteractions")
    finish!(pbar)
    return BlockSparseMatrix(blocks, values, nearvalues, size(assembler))
end

function _blocksparseassemble(
    assembler::H2Factory.AbstractNearInteractionsAssembler{T},
    selfvalues,
    values,
    nearvalues,
    verbose::Bool;
) where {T}
    # Keep self-interaction blocks separate from the off-diagonal blocks.
    selfblocks = Vector{Matrix{T}}(undef, length(selfvalues))
    addblocks!(selfblocks, selfvalues, selfvalues)
    nonselfblocks = Vector{Matrix{T}}(undef, length(values))
    addblocks!(nonselfblocks, values, nearvalues)
    pbar = progressbar(
        length(selfblocks) + length(nonselfblocks) + 2,
        verbose;
        desc="Computing self interactions",
    )

    primer = assembleblockprimer(assembler)

    Threads.@threads :static for i in eachindex(selfblocks)
        assembleblock(
            assembler,
            selfblocks[i],
            selfvalues[i],
            selfvalues[i],
            blockassembler(assembler; primer=primer),
        )

        next!(pbar)
    end
    next!(pbar; desc="Computing nonself interactions")

    Threads.@threads for i in eachindex(nonselfblocks)
        assembleblock(
            assembler,
            nonselfblocks[i],
            values[i],
            nearvalues[i],
            blockassembler(assembler; primer=primer),
        )
        next!(pbar)
    end
    next!(pbar; desc="Finished computation of nearinteractions")
    finish!(pbar)

    return SymmetricBlockMatrix(
        selfblocks, selfvalues, nonselfblocks, values, nearvalues, size(assembler)
    )
end

function addblocks!(blocks::Vector{Matrix{T}}, values, nearvalues) where {T}
    @assert length(blocks) == length(values) == length(nearvalues)
    Threads.@threads for i in eachindex(values)
        blocks[i] = zeros(T, length(values[i]), length(nearvalues[i]))
    end
    return blocks
end
