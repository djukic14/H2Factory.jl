# H2 map hierarchy and top-level `LinearMaps` multiplication entry points.

"""
    AbstractH2Map{T} <: LinearMap{T}

Abstract base type for H2 linear maps with element type `T`.
"""
abstract type AbstractH2Map{T} <: LinearMap{T} end

"""
    GalerkinH2Map{T} <: AbstractH2Map{T}

Abstract H2 map whose test and trial sides share the same space and tree.
"""
abstract type GalerkinH2Map{T} <: AbstractH2Map{T} end

"""
    PetrovGalerkinH2Map{T} <: AbstractH2Map{T}

Abstract H2 map with distinct test and trial spaces or trees.
"""
abstract type PetrovGalerkinH2Map{T} <: AbstractH2Map{T} end

function maptrait(::Type{<:AbstractH2Map{T}}) where {T}
    return isForward()
end

const AbstractH2Maps{T} = Union{
    <:AbstractH2Map{T},
    LinearMaps.AdjointMap{T,<:AbstractH2Map{T}},
    LinearMaps.TransposeMap{T,<:AbstractH2Map{T}},
}

const AbstractGalerkinMaps{T} = Union{
    <:GalerkinH2Map{T},
    LinearMaps.AdjointMap{T,<:GalerkinH2Map{T}},
    LinearMaps.TransposeMap{T,<:GalerkinH2Map{T}},
}

# Standard multiplication uses the same implementation as diagnostic multiplication, with the
# no-op diagnostics policy.
function LinearMaps._unsafe_mul!(y::AbstractVector, A::AbstractH2Maps, x::AbstractVector)
    return _unsafe_mul!(y, A, x, NoDiagnostics())
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::AbstractH2Maps, x::AbstractVector, α::Number, β::Number
)
    return _unsafe_mul!(y, A, x, α, β, NoDiagnostics())
end

# Preserve the concrete diagnostics type so the disabled path inlines. Workspace ownership is taken
# once around the complete near/far operation; the far pass therefore calls unguarded `_farmul!`.
function _unsafe_mul!(
    y::AbstractVector, A::AbstractH2Maps{T}, x::AbstractVector, diagnostics::D
) where {T,D<:AbstractDiagnostics}
    return withworkspace(A) do
        # Record whole-operation totals; the far pass may additionally record individual levels.
        with_timing(diagnostics, :near, :total; mode=diagnosticmode(A)) do
            return nearmul!(y, A, x)
        end
        with_timing(diagnostics, :far, :total; mode=diagnosticmode(A)) do
            return _farmul!(y, A, x, one(T), one(T), diagnostics)
        end
        return y
    end
end

function _unsafe_mul!(
    y::AbstractVector,
    A::AbstractH2Maps{T},
    x::AbstractVector,
    α::Number,
    β::Number,
    diagnostics::D,
) where {T,D<:AbstractDiagnostics}
    return withworkspace(A) do
        with_timing(diagnostics, :near, :total; mode=diagnosticmode(A)) do
            return nearmul!(y, A, x, α, β)
        end
        with_timing(diagnostics, :far, :total; mode=diagnosticmode(A)) do
            return _farmul!(y, A, x, α, 1, diagnostics)
        end
        return y
    end
end

function Base.size(A::AbstractH2Map)
    return size(nearinteractions(A))
end

function Base.size(A::AbstractH2Map, dim)
    return size(nearinteractions(A), dim)
end

function Base.eltype(A::AbstractH2Map{T}) where {T}
    return T
end
