"""
    PassThroughNearInteractionsAssembler{T}(operator, testspace, trialspace, nearinteractions)

Return an already assembled near-interaction matrix unchanged. The operator and spaces are accepted
for the common assembler constructor shape but are not used. Calling this assembler does not walk
the tree. See [`PassThroughNearInteractionsStrategy`](@ref).
"""
struct PassThroughNearInteractionsAssembler{T,N} <: AbstractNearInteractionsAssembler{T}
    nearinteractions::N

    function PassThroughNearInteractionsAssembler{T}(
        ::Any, ::Any, ::Any, nearinteractions
    ) where {T}
        return new{T,typeof(nearinteractions)}(nearinteractions)
    end
end

"""
    nearinteractions(assembler::PassThroughNearInteractionsAssembler)

The stored, already-computed near-interaction matrix.
"""
nearinteractions(assembler::PassThroughNearInteractionsAssembler) =
    assembler.nearinteractions

function (f::PassThroughNearInteractionsAssembler)(::Any, ::Any; kwargs...)
    return f.nearinteractions
end
