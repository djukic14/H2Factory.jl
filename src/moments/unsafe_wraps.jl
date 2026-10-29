"""
    wrapvector(m::AbstractMoment{D,T}, i) where {D,T}

Return a vector view of component `i`. Dense moments reuse a cached view; other moments
reshape the component array.
"""
function wrapvector(m::AbstractMoment{D,T}, i) where {D,T<:DenseArray}
    return m.vectors[i]
end

function wrapvector(m::AbstractMoment{D,T}, i) where {D,T}
    return reshape(m[i], :)
end

"""
    alignoffset(offset, alignment)

Return the bytes needed to reach the next aligned address, or zero if already aligned.
"""
alignoffset(offset, alignment) = (alignment - offset % alignment) % alignment

# Advance `p` to the next `alignment`-byte boundary.
_align(p::Ptr, alignment=MOMENTALIGNMENT) = p + alignoffset(UInt64(p), alignment)

"""
    MOMENTALIGNMENT

Byte alignment for ordinary moment arrays.
"""
const MOMENTALIGNMENT = 16

"""
    TOEPLITZALIGNMENT

Byte alignment for padded Toeplitz arrays. FFT plans require the input to have a compatible
alignment class; 64 bytes covers supported FFTW SIMD classes. Ordinary moment arrays are not
passed to those plans.
"""
const TOEPLITZALIGNMENT = 64

"""
    alignmentslack(moment) -> Int

Return the worst-case alignment padding for `moment`, in elements. Buffer allocation adds this to
`numberofentries(moment)` so the last wrapped array remains inside the pool.
"""
function alignmentslack(moment::AbstractMoment)
    T = eltype(eltype(typeof(moment)))
    percomponent = cld(MOMENTALIGNMENT - 1, sizeof(T))
    return length(eachindex(moment)) * percomponent
end

# Budget alignment for both ordinary and padded arrays.
function alignmentslack(moment::AbstractToeplitzMoment)
    T = eltype(eltype(typeof(moment)))
    percomponent =
        cld(MOMENTALIGNMENT - 1, sizeof(T)) + cld(TOEPLITZALIGNMENT - 1, sizeof(T))
    return length(eachindex(moment)) * percomponent
end

function wrapbuffer(buffer, currentpointer, MomentType, momentsize)
    @assert eltype(buffer) === eltype(eltype(MomentType)) "buffer element type must match " *
        "moment element type, got $(eltype(buffer)) and $(eltype(eltype(MomentType)))"

    momentarrays = Vector{Array{eltype(eltype(MomentType)),length(momentsize)}}(
        undef, dim(MomentType)
    )

    for i in eachindex(momentarrays)
        currentpointer = _align(currentpointer)
        momentarrays[i] = unsafe_wrap(eltype(momentarrays), currentpointer, momentsize)
        currentpointer += prod(momentsize) * sizeof(eltype(buffer))
    end

    return currentpointer, MomentType(SVector{dim(MomentType)}(momentarrays))
end

function wrapbuffer(
    buffer, currentpointer, MomentType::Type{<:AbstractToeplitzMoment}, momentsize
)
    @assert eltype(buffer) === eltype(eltype(MomentType)) "buffer element type must match " *
        "moment element type, got $(eltype(buffer)) and $(eltype(eltype(MomentType)))"

    momentsize, tsize = momentsize
    momentarrays = Vector{Array{eltype(eltype(MomentType)),length(momentsize)}}(
        undef, dim(MomentType)
    )

    toeplitzmomentarrays = Vector{Array{eltype(eltype(MomentType)),length(momentsize)}}(
        undef, dim(MomentType)
    )

    for i in eachindex(momentarrays)
        currentpointer = _align(currentpointer)
        momentarrays[i] = unsafe_wrap(eltype(momentarrays), currentpointer, momentsize)
        currentpointer += prod(momentsize) * sizeof(eltype(buffer))
    end

    for i in eachindex(toeplitzmomentarrays)
        # Match the alignment class required by the FFT plan.
        currentpointer = _align(currentpointer, TOEPLITZALIGNMENT)
        toeplitzmomentarrays[i] = unsafe_wrap(
            eltype(toeplitzmomentarrays), currentpointer, tsize
        )
        currentpointer += prod(tsize) * sizeof(eltype(buffer))
    end

    return currentpointer,
    MomentType(
        SVector{dim(MomentType)}(momentarrays),
        SVector{dim(MomentType)}(toeplitzmomentarrays),
    )
end
