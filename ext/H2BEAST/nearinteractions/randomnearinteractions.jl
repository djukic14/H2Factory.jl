function assembleblock(
    assembler::H2Factory.RandomNearInteractionsAssembler, I, J, V, tdata, sdata, blkasm
)
    for m in eachindex(tdata), n in eachindex(sdata)
        I[(n - 1) * length(tdata) + m] = tdata[m]
        J[(n - 1) * length(tdata) + m] = sdata[n]
        V[(n - 1) * length(tdata) + m] = randn(Base.eltype(V))
    end

    return nothing
end

function blockassembler(assembler::H2Factory.RandomNearInteractionsAssembler; kwargs...)
    return nothing
end
