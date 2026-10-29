function _sparseassemble(
    assembler::H2Factory.AbstractNearInteractionsAssembler{T},
    values,
    nearvalues,
    verbose::Bool;
) where {T}
    # Prefix sums give each block a separate slice of the sparse triplet buffers.
    numberofcoefficients = Vector{Int}(undef, length(values) + 1)
    numberofcoefficients[begin] = 0

    Threads.@threads for i in eachindex(values)
        numberofcoefficients[i + 1] = length(values[i]) * length(nearvalues[i])
    end
    cumsum!(numberofcoefficients, numberofcoefficients)

    I = Vector{Int}(undef, numberofcoefficients[end])
    J = Vector{Int}(undef, numberofcoefficients[end])
    V = zeros(T, numberofcoefficients[end])

    primer = assembleblockprimer(assembler)

    pbar = progressbar(length(values) + 2, verbose; desc="Computing nearinteractions")

    Threads.@threads :static for i in eachindex(values)
        assembleblock(
            assembler,
            view(I, (numberofcoefficients[i] + 1):(numberofcoefficients[i + 1])),
            view(J, (numberofcoefficients[i] + 1):(numberofcoefficients[i + 1])),
            view(V, (numberofcoefficients[i] + 1):(numberofcoefficients[i + 1])),
            values[i],
            nearvalues[i],
            blockassembler(assembler; primer=primer),
        )
        next!(pbar)
    end

    next!(pbar; desc="Computing sparse representation")
    nearinteractions = SparseArrays.sparse(I, J, V, size(assembler, 1), size(assembler, 2))
    next!(pbar; desc="Finished computation of nearinteractions")
    finish!(pbar)

    return nearinteractions
end
