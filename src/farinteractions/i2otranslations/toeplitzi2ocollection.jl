"""
    ToeplitzI2OTranslationCollection(collection, tree)

Read-only Toeplitz couplings and the tree used to resolve translation levels. FFT plans belong to
the workspace-local [`ToeplitzI2OChannel`](@ref).

Uses [`ToeplitzModulation`](@ref); combining it with [`NoModulation`](@ref) raises `ArgumentError`.
"""
struct ToeplitzI2OTranslationCollection{I,T} <: AbstractI2OTranslationCollection
    i2ocollection::I
    tree::T
end

function Base.getindex(T::ToeplitzI2OTranslationCollection, key)
    return T.i2ocollection[key]
end

function i2otrait(T::ToeplitzI2OTranslationCollection)
    return i2otrait(T.i2ocollection)
end

modulation(::ToeplitzI2OTranslationCollection) = ToeplitzModulation()

function I2OTranslation(::ToeplitzI2OTranslationCollection, ::NoModulation)
    return throw(ArgumentError("Toeplitz couplings require ToeplitzModulation()"))
end

# Rejects ANY moment -- not only a Toeplitz one -- paired with `NoModulation` through this BARE
# collection type. This is the original guard: it catches a mismatched moment type reaching a
# Toeplitz coupling directly, which the moment-typed rejection in `i2oprotocol.jl` (added so the
# same rejection also holds through an arbitrary WRAPPER around this collection, which this
# coupling-typed dispatch alone can never see) does not cover on its own.
function modulate_in!(
    moment,
    level::Int,
    maptrait,
    ::NoModulation,
    ::ToeplitzI2OTranslationCollection,
    channel,
    scheduler,
    buffer,
)
    return throw(ArgumentError("Toeplitz couplings require ToeplitzModulation()"))
end
function modulate_out!(
    moment,
    level::Int,
    maptrait,
    ::NoModulation,
    ::ToeplitzI2OTranslationCollection,
    channel,
    scheduler,
    buffer,
)
    return throw(ArgumentError("Toeplitz couplings require ToeplitzModulation()"))
end

# Intersection methods: a Toeplitz MOMENT reaching this BARE collection type matches both the
# moment-typed rejection above (in `i2oprotocol.jl`) and the coupling-typed one just above here,
# and neither is strictly more specific than the other -- Julia would otherwise raise a method
# ambiguity. Same outcome either way; these exist only so there is one, unambiguous, most-specific
# method to dispatch to.
function modulate_in!(
    moment::AbstractToeplitzMoment,
    level::Int,
    maptrait,
    ::NoModulation,
    ::ToeplitzI2OTranslationCollection,
    channel,
    scheduler,
    buffer,
)
    return throw(ArgumentError("Toeplitz couplings require ToeplitzModulation()"))
end
function modulate_out!(
    moment::AbstractToeplitzMoment,
    level::Int,
    maptrait,
    ::NoModulation,
    ::ToeplitzI2OTranslationCollection,
    channel,
    scheduler,
    buffer,
)
    return throw(ArgumentError("Toeplitz couplings require ToeplitzModulation()"))
end

function H2Trees.level(T::ToeplitzI2OTranslationCollection, translatingnode::Int)
    return H2Trees.level(H2Trees.tree(T), translatingnode)
end

function H2Trees.tree(T::ToeplitzI2OTranslationCollection)
    return T.tree
end

# Symmetry metadata belongs to the wrapped collection and must be forwarded.
assertnosymmetry(tfs::ToeplitzI2OTranslationCollection) =
    assertnosymmetry(tfs.i2ocollection)

translationsymmetrygroup(tfs::ToeplitzI2OTranslationCollection) =
    translationsymmetrygroup(tfs.i2ocollection)

function translationsymmetry(
    tfs::ToeplitzI2OTranslationCollection, receivingnode::Int, translatingnode::Int
)
    return translationsymmetry(tfs.i2ocollection, receivingnode, translatingnode)
end

# Forward and transpose share this kernel. The channel swaps FFT plans for transpose.
function applycouplingkernel!(
    out,
    in,
    tfs::ToeplitzI2OTranslationCollection,
    ::ToeplitzI2OTranslationTrait,
    receivingnode::Int,
    translatingnode::Int,
    direction::Union{ForwardTranslation,TransposeTranslation},
    ::Overwrite,
    context::TranslationContext,
)
    toeplitzmul!(
        out,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        in,
        context.scheduler,
    )
    return out
end

function applycouplingkernel!(
    out,
    in,
    tfs::ToeplitzI2OTranslationCollection,
    ::ToeplitzI2OTranslationTrait,
    receivingnode::Int,
    translatingnode::Int,
    direction::Union{ForwardTranslation,TransposeTranslation},
    ::Accumulate,
    context::TranslationContext,
)
    toeplitzmul!(
        out,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        in,
        true,
        true,
        context.scheduler,
    )
    return out
end

# Adjoint uses the swapped node pair, conjugated coefficients, and forward-order FFT plans.
function applycouplingkernel!(
    out,
    in,
    tfs::ToeplitzI2OTranslationCollection,
    ::ToeplitzI2OTranslationTrait,
    receivingnode::Int,
    translatingnode::Int,
    direction::AdjointTranslation,
    ::Overwrite,
    context::TranslationContext,
)
    adjointtoeplitzmul!(
        out,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        in,
        context.scheduler,
    )
    return out
end

function applycouplingkernel!(
    out,
    in,
    tfs::ToeplitzI2OTranslationCollection,
    ::ToeplitzI2OTranslationTrait,
    receivingnode::Int,
    translatingnode::Int,
    direction::AdjointTranslation,
    ::Accumulate,
    context::TranslationContext,
)
    adjointtoeplitzmul!(
        out,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        in,
        true,
        true,
        context.scheduler,
    )
    return out
end

"""
    toeplitzi2ocollection(tree, translatingplan, translationtrait, Translator; kwargs...)

Build a read-only Toeplitz coupling collection without allocating workspace FFT plans.
Implemented by the `H2Toeplitz` extension. Use [`i2ochannels`](@ref) to build its workspace channel.
"""
function toeplitzi2ocollection end

"""
    ToeplitzI2OChannel(fftplans, ifftplans, levels)

Workspace-local FFT plan pools indexed by level and chunk. `fftplans` and `ifftplans` contain
one pool per entry in `levels`. Borrow a plan pair with
`borrowbuffer(channel, level, chunkid, maptrait)`.

Transpose swaps the forward and inverse plans; adjoint uses the forward plan order.
"""
struct ToeplitzI2OChannel{F,I}
    fftplans::F
    ifftplans::I
end

function borrowbuffer(channel::ToeplitzI2OChannel, level::Int, chunkid::Int, maptrait)
    !haskey(channel.fftplans, level) &&
        return (fftplan=EmptyChannels(), ifftplan=EmptyChannels())
    return (
        fftplan=takefftplan!(channel, level, chunkid, maptrait),
        ifftplan=takeifftplan!(channel, level, chunkid, maptrait),
    )
end

function ToeplitzI2OChannel(fftplans, ifftplans, levels)
    @assert length(fftplans) == length(ifftplans) == length(levels)
    isempty(fftplans) && return ToeplitzI2OChannel([], [])

    fftchannels = []
    ifftchannels = []
    for i in eachindex(fftplans)
        push!(fftchannels, createchannel(fftplans[i]))
        push!(ifftchannels, createchannel(ifftplans[i]))
    end
    fftchannels = convert(Vector{typeof(fftchannels[1])}, fftchannels)
    ifftchannels = convert(Vector{typeof(ifftchannels[1])}, ifftchannels)

    return ToeplitzI2OChannel(
        Dict(zip(levels, fftchannels)), Dict(zip(levels, ifftchannels))
    )
end

function _takeifftplan!(channel::ToeplitzI2OChannel, level::Int, chunkid::Int)
    return borrowbuffer(channel.ifftplans[level], chunkid)
end

# Forward: ifft * D * fft. Transpose swaps the plans: fft * D * ifft.
# Adjoint keeps forward plan order and conjugates D in `adjointtoeplitzmul!`.
function takeifftplan!(
    channel::ToeplitzI2OChannel, level::Int, chunkid::Int, ::Union{isForward,isAdjoint}
)
    return _takeifftplan!(channel, level, chunkid)
end

function takeifftplan!(channel::ToeplitzI2OChannel, level::Int, chunkid::Int, ::isTranspose)
    return _takefftplan!(channel, level, chunkid)
end

function _takefftplan!(channel::ToeplitzI2OChannel, level::Int, chunkid::Int)
    return borrowbuffer(channel.fftplans[level], chunkid)
end

function takefftplan!(
    channel::ToeplitzI2OChannel, level::Int, chunkid::Int, ::Union{isForward,isAdjoint}
)
    return _takefftplan!(channel, level, chunkid)
end

function takefftplan!(channel::ToeplitzI2OChannel, level::Int, chunkid::Int, ::isTranspose)
    return _takeifftplan!(channel, level, chunkid)
end
