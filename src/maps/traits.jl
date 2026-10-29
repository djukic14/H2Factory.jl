# Map orientation traits and forward-map unwrapping. Accessors use these to reach the storage behind
# `transpose(A)` and `adjoint(A)` wrappers.

abstract type AbstractMapTrait end

struct isForward <: AbstractMapTrait end
struct isTranspose <: AbstractMapTrait end
struct isAdjoint <: AbstractMapTrait end

"""
    maptrait(A)
    maptrait(::Type{<:AbstractH2Map}) -> AbstractMapTrait

Return the map orientation as `isForward()`, `isTranspose()`, or `isAdjoint()`.

Custom map types should implement the type-level method. The value form delegates to
`maptrait(typeof(A))`; transpose and adjoint wrappers are handled automatically.
"""
function maptrait(A)
    return maptrait(typeof(A))
end

# Terminate the value-to-type forwarding for undeclared types. Without this fallback,
# `maptrait(DataType)` recurses because `typeof(DataType) === DataType`. More specific map methods
# still take precedence.
function maptrait(::Type{T}) where {T}
    return throw(
        ArgumentError(
            "maptrait is not declared for $(T). Define `H2Factory.maptrait(::Type{<:$(T)})` " *
            "for a map type, or pass the map rather than one of its components.",
        ),
    )
end

function maptrait(::Type{LinearMaps.AdjointMap{T,M}}) where {T,M}
    return isAdjoint()
end

function maptrait(::Type{LinearMaps.TransposeMap{T,M}}) where {T,M}
    return isTranspose()
end

function forwardmap(A::Union{LinearMaps.AdjointMap,LinearMaps.TransposeMap})
    return A.lmap
end

function forwardmap(A)
    return forwardmap(A, maptrait(A))
end

function forwardmap(A, ::isForward)
    return A
end
