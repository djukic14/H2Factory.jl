"""
    ToeplitzMatrixRepresentation(toeplitz)

Wrap a Toeplitz coupling stored as a padded, FFT-domain array. The multiply kernels read
`toeplitz` elementwise against padded moment arrays.
"""
struct ToeplitzMatrixRepresentation{T} <: LinearMap{T}
    toeplitz::T
end

struct AdjointToeplitzMatrixRepresentation{T} <: LinearMap{T}
    toeplitz::T
end

function Base.adjoint(T::ToeplitzMatrixRepresentation)
    return AdjointToeplitzMatrixRepresentation(T.toeplitz)
end

function LinearAlgebra.Adjoint(T::ToeplitzMatrixRepresentation)
    return AdjointToeplitzMatrixRepresentation(T.toeplitz)
end

function Base.eltype(
    T::Union{ToeplitzMatrixRepresentation,AdjointToeplitzMatrixRepresentation}
)
    return eltype(T.toeplitz)
end

function Base.eltype(
    ::Union{
        Type{ToeplitzMatrixRepresentation{T}},Type{AdjointToeplitzMatrixRepresentation{T}}
    },
) where {T}
    return eltype(T)
end

function toeplitzeltype(
    T::Union{ToeplitzMatrixRepresentation,AdjointToeplitzMatrixRepresentation}
)
    return typeof(T.toeplitz)
end

function toeplitzeltype(
    ::Union{
        Type{ToeplitzMatrixRepresentation{T}},Type{AdjointToeplitzMatrixRepresentation{T}}
    },
) where {T}
    return T
end

function toeplitzeltype(t::AbstractArray)
    return toeplitzeltype(typeof(t))
end

function toeplitzeltype(::Type{<:AbstractArray{T}}) where {T}
    return toeplitzeltype(T)
end

function Base.length(
    T::Union{ToeplitzMatrixRepresentation,AdjointToeplitzMatrixRepresentation}
)
    return length(T.toeplitz)
end

function Base.size(
    T::Union{ToeplitzMatrixRepresentation,AdjointToeplitzMatrixRepresentation}
)
    return length.(axes(T))
end

function toeplitzsize(
    tf::Union{ToeplitzMatrixRepresentation,AdjointToeplitzMatrixRepresentation}
)
    return size(tf.toeplitz)
end

function toeplitzsize(
    tf::AbstractArray{
        <:Union{ToeplitzMatrixRepresentation,AdjointToeplitzMatrixRepresentation}
    },
)
    return size(first(tf).toeplitz)
end

# Zero the padded FFT arrays, not the ordinary moment arrays.
function _zerotoeplitz!(out)
    for i in eachindex(out)
        toeplitzmoment(out, i) .= 0
    end
    return out
end

# Accumulate the elementwise product. Callers clear `out` first when overwriting.
# `transform` is `identity` for a plain operator and `conj` for its adjoint.
function _accumulatetoeplitz!(tout, localtf, tin, transform::F) where {F}
    @inbounds @simd ivdep for k in eachindex(localtf, tin, tout)
        tout[k] += transform(localtf[k]) * tin[k]
    end
    return tout
end
_accumulatetoeplitz!(tout, localtf, tin) =
    _accumulatetoeplitz!(tout, localtf, tin, identity)

# Supported coupling shapes: one operator, a vector, or a matrix of operators.
const _ToeplitzCoupling = Union{
    ToeplitzMatrixRepresentation,
    Vector{<:ToeplitzMatrixRepresentation},
    Matrix{<:ToeplitzMatrixRepresentation},
}

"""
    toeplitzmul!(out, coupling, in, threading)
    toeplitzmul!(out, coupling, in, α::Bool, β::Bool, threading)

Multiply padded Toeplitz moments in the FFT domain. `coupling` may be one
[`ToeplitzMatrixRepresentation`](@ref), a vector, or a matrix of them. With `α` and `β`,
compute `out = α * coupling * in + β * out`.
"""
toeplitzmul!(out, tf::_ToeplitzCoupling, in, threading::Scheduler) =
    _toeplitzmul!(out, tf, in, identity, threading)
function toeplitzmul!(
    out, tf::_ToeplitzCoupling, in, α::Bool, β::Bool, threading::Scheduler
)
    return _toeplitzmul!(out, tf, in, identity, α, β, threading)
end
"""
    adjointtoeplitzmul!(out, coupling, in, threading)
    adjointtoeplitzmul!(out, coupling, in, α::Bool, β::Bool, threading)

Apply the adjoint Toeplitz coupling in the FFT domain. Uses the same accumulation rule as
[`toeplitzmul!`](@ref), with the coupling entries conjugated.
"""
function adjointtoeplitzmul!(out, tf::_ToeplitzCoupling, in, threading::Scheduler)
    return _toeplitzmul!(out, tf, in, conj, threading)
end
function adjointtoeplitzmul!(
    out, tf::_ToeplitzCoupling, in, α::Bool, β::Bool, threading::Scheduler
)
    return _toeplitzmul!(out, tf, in, conj, α, β, threading)
end

# out = β*out + α*(transform(tf) .* in), componentwise: one translation operator per component.
function _toeplitzmul!(
    out, tf::ToeplitzMatrixRepresentation, in, transform::F, α::Bool, β::Bool, ::Scheduler
) where {F}
    β || _zerotoeplitz!(out)
    α || return out

    for i in eachindex(out)
        _accumulatetoeplitz!(
            toeplitzmoment(out, i), tf.toeplitz, toeplitzmoment(in, i), transform
        )
    end
    return out
end

# out[i] = β*out[i] + α*Σ_j transform(tf[i,j]) .* in[j]: a full D×D block of translation operators.
function _toeplitzmul!(
    out::M1,
    tf::Matrix{<:ToeplitzMatrixRepresentation},
    in::M2,
    transform::F,
    α::Bool,
    β::Bool,
    ::Scheduler,
) where {D,TM1,TM2,F,M1<:AbstractToeplitzMoment{D,TM1},M2<:AbstractToeplitzMoment{D,TM2}}
    β || _zerotoeplitz!(out)
    α || return out

    for i in eachindex(out)
        tout = toeplitzmoment(out, i)
        for j in eachindex(in)
            _accumulatetoeplitz!(tout, tf[i, j].toeplitz, toeplitzmoment(in, j), transform)
        end
    end
    return out
end

# A vector coupling maps between one component and several. A matrix coupling handles
# cross-component products when both sides have several components.
function _toeplitzmul!(
    out::M1,
    tf::Vector{<:ToeplitzMatrixRepresentation},
    in::M2,
    transform::F,
    α::Bool,
    β::Bool,
    ::Scheduler,
) where {
    D1,D2,TM1,TM2,F,M1<:AbstractToeplitzMoment{D1,TM1},M2<:AbstractToeplitzMoment{D2,TM2}
}
    # Checked before `out` is touched, so an unsupported shape cannot leave a half-zeroed moment.
    if D1 != 1 && D2 != 1
        error(
            "This operation with D1 = $D1, D2 = $D2 α = $α and β = $β is not supported. Either $D1 or $D2 has to be 1.",
        )
    end

    β || _zerotoeplitz!(out)
    α || return out

    if D1 == 1
        tout = toeplitzmoment(out, 1)
        for i in eachindex(in)
            _accumulatetoeplitz!(tout, tf[i].toeplitz, toeplitzmoment(in, i), transform)
        end
    else
        tin = toeplitzmoment(in, 1)
        for i in eachindex(out)
            _accumulatetoeplitz!(toeplitzmoment(out, i), tf[i].toeplitz, tin, transform)
        end
    end
    return out
end

function _toeplitzmul!(
    out, tf::ToeplitzMatrixRepresentation, in, transform::F, ::Scheduler
) where {F}
    for i in eachindex(out)
        tout = toeplitzmoment(out, i)
        tin = toeplitzmoment(in, i)
        localtf = tf.toeplitz
        @inbounds @simd ivdep for k in eachindex(localtf, tin, tout)
            tout[k] = transform(localtf[k]) * tin[k]
        end
    end
    return out
end

function _toeplitzmul!(
    out::M1, tf::Matrix{<:ToeplitzMatrixRepresentation}, in::M2, transform::F, ::Scheduler
) where {D,TM1,TM2,F,M1<:AbstractToeplitzMoment{D,TM1},M2<:AbstractToeplitzMoment{D,TM2}}
    for i in eachindex(out)
        toeplitzmoment(out, i) .= 0
    end

    for i in eachindex(out)
        tout = toeplitzmoment(out, i)
        for j in eachindex(in)
            tin = toeplitzmoment(in, j)
            localtf = tf[i, j].toeplitz
            @inbounds @simd ivdep for k in eachindex(localtf, tin, tout)
                tout[k] += transform(localtf[k]) * tin[k]
            end
        end
    end
    return out
end

function _toeplitzmul!(
    out::M1, tf::Vector{<:ToeplitzMatrixRepresentation}, in::M2, transform::F, ::Scheduler
) where {
    D1,D2,TM1,TM2,F,M1<:AbstractToeplitzMoment{D1,TM1},M2<:AbstractToeplitzMoment{D2,TM2}
}
    if D1 == 1
        for i in eachindex(in)
            if i == 1
                tout = toeplitzmoment(out, 1)
                tin = toeplitzmoment(in, i)
                localtf = tf[i].toeplitz
                @inbounds @simd ivdep for k in eachindex(localtf, tin, tout)
                    tout[k] = transform(localtf[k]) * tin[k]
                end
            else
                tout = toeplitzmoment(out, 1)
                tin = toeplitzmoment(in, i)
                localtf = tf[i].toeplitz
                @inbounds @simd ivdep for k in eachindex(localtf, tin, tout)
                    tout[k] += transform(localtf[k]) * tin[k]
                end
            end
        end
    elseif D2 == 1
        for i in eachindex(out)
            tout = toeplitzmoment(out, i)
            tin = toeplitzmoment(in, 1)
            localtf = tf[i].toeplitz
            @inbounds @simd ivdep for k in eachindex(localtf, tin, tout)
                tout[k] = transform(localtf[k]) * tin[k]
            end
        end
    else
        error(
            "This operation with D1 = $D1 and D2 = $D2 is not supported. Either $D1 or $D2 has to be 1.",
        )
    end
    return out
end

function _toeplitzmodulate_in!(in::AbstractToeplitzMoment, buffer)
    fftplan = buffer.fftplan
    for i in eachindex(in)
        toeplitzmoment(in, i) .= 0
        fftplan * _copytobuffer!(toeplitzmoment(in, i), in[i])
    end
    return in
end

function _toeplitzmodulate_out!(out::AbstractToeplitzMoment, buffer)
    ifftplan = buffer.ifftplan
    for i in eachindex(out)
        _copytomoment!(out[i], ifftplan * toeplitzmoment(out, i))
    end
    return out
end

# Copy the unpadded array to or from the leading corner of the FFT buffer, at any rank.
function _copytobuffer!(buffer, x::Array{T,N}) where {T,N}
    view(buffer, axes(x)...) .= x
    return buffer
end

function _copytomoment!(out::Array{T,N}, buffer) where {T,N}
    return out .= view(buffer, axes(out)...)
end

function Base.axes(
    T::Union{ToeplitzMatrixRepresentation,AdjointToeplitzMatrixRepresentation}
)
    ax = Base.OneTo(prod(floor.(Int, (size(T.toeplitz) .+ 1) ./ 2)))
    return ax, ax
end
