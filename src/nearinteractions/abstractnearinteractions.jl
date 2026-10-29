"""
    AbstractNearInteractionsAssembler{T}

Base type for near-interaction assemblers with scalar type `T`. Concrete assemblers implement
[`assemble`](@ref) for the interaction groups produced by H2Trees.
"""
abstract type AbstractNearInteractionsAssembler{T} end

function isgalerkinsymmetric(operator)
    return isgalerkinsymmetric(typeof(operator))
end

function isgalerkinsymmetric(::Type{T}) where {T}
    return false
end

"""
    supportsgalerkinsymmetric(assembler) -> Bool

Return whether `assembler` can use symmetric Galerkin assembly for its operator.
"""
function supportsgalerkinsymmetric(::Type{T}) where {T}
    return false
end

function supportsgalerkinsymmetric(assembler::AbstractNearInteractionsAssembler)
    return supportsgalerkinsymmetric(typeof(assembler)) &&
           isgalerkinsymmetric(operator(assembler))
end

function (assembler::AbstractNearInteractionsAssembler)(
    tree, isnear; verbose::Bool=false, kwargs...
)
    return assembler(tree, H2Trees.treetrait(tree), isnear; verbose=verbose, kwargs...)
end

function (assembler::AbstractNearInteractionsAssembler)(
    tree, ::H2Trees.isBlockTree, isnear; verbose::Bool=false, kwargs...
)
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false, kwargs...
    )
    return assemble(assembler, values, nearvalues, verbose)
end

function (assembler::AbstractNearInteractionsAssembler)(
    tree,
    ::H2Trees.AbstractTreeTrait,
    isnear;
    verbose::Bool=false,
    supportsgalerkinsymmetric=H2Factory.supportsgalerkinsymmetric(assembler),
    kwargs...,
)
    if supportsgalerkinsymmetric
        selfvalues, values, nearvalues = H2Trees.nearinteractions(
            tree;
            isnear=GalerkinSymmetricIsNearFunctor(isnear),
            extractselfvalues=true,
            kwargs...,
        )
        return assemble(assembler, selfvalues, values, nearvalues, verbose)
    else
        values, nearvalues = H2Trees.nearinteractions(
            tree; isnear=isnear, extractselfvalues=false, kwargs...
        )
        return assemble(assembler, values, nearvalues, verbose)
    end
end

# `assemble` has no abstract fallback. An incomplete assembler must fail with a `MethodError`.

function Base.size(assembler::AbstractNearInteractionsAssembler, dim::Int)
    return size(assembler)[dim]
end

function operator(assembler::AbstractNearInteractionsAssembler)
    return assembler.operator
end

function trialspace(assembler::AbstractNearInteractionsAssembler)
    return assembler.trialspace
end

function testspace(assembler::AbstractNearInteractionsAssembler)
    return assembler.testspace
end

function quadstrategy(assembler::AbstractNearInteractionsAssembler)
    return assembler.quadstrat
end

struct GalerkinSymmetricIsNearFunctor{N}
    isnear::N
end

function (f::GalerkinSymmetricIsNearFunctor)(tree, nodea, nodeb)
    if H2Trees.isleaf(tree, nodeb)
        return f.isnear(tree, nodea, nodeb) && nodea > nodeb
    else
        return f.isnear(tree, nodea, nodeb)
    end
end
