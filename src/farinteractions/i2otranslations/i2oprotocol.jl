# Convert each node's basis once; apply couplings for each interacting node pair.

# Protocol types

"""
    AbstractI2OCouplingCollection

A collection of I2O couplings keyed by node pair. Basis conversion belongs to
[`AbstractI2OModulation`](@ref).

Concrete collections implement:

  - [`assertnosymmetry`](@ref): reject unhandled symmetry tags, forward to a wrapped collection,
    or return `nothing` if no tags can remain.
  - [`i2otrait`](@ref): select the coupling-storage trait.
  - [`applycouplingkernel!`](@ref): apply a pair's coupling for each supported direction and
    accumulation mode.

Do not specialize [`applycoupling!`](@ref): it checks symmetry before calling the kernel.
An undeclared `assertnosymmetry` method raises an error on first use.
"""
abstract type AbstractI2OCouplingCollection end

"""
    AbstractI2OModulation

A policy for converting moments into the coupling basis and back. Concrete policies implement
the eight-argument [`modulate_in!`](@ref) and [`modulate_out!`](@ref) methods.
"""
abstract type AbstractI2OModulation end

"""
    NoModulation()

Leave moments unchanged when they are already in the coupling basis.
Toeplitz coupling collections reject this policy; use [`ToeplitzModulation`](@ref) instead.
"""
struct NoModulation <: AbstractI2OModulation end

"""
    ToeplitzModulation()

Convert Toeplitz moments to and from the frequency domain used by a Toeplitz coupling.
"""
struct ToeplitzModulation <: AbstractI2OModulation end

"""
    I2OTranslation(coupling, modulation = modulation(coupling))

Combine an [`AbstractI2OCouplingCollection`](@ref) with an [`AbstractI2OModulation`](@ref) policy.
Maps return this object from [`i2otranslator`](@ref); scratch channels belong to their workspaces.

[`coupling`](@ref) and [`modulation`](@ref) return the two parts. Construction checks their types;
the required coupling methods are checked when the translation is used.
"""
struct I2OTranslation{C<:AbstractI2OCouplingCollection,M<:AbstractI2OModulation}
    coupling::C
    modulation::M
end

I2OTranslation(coupling) = I2OTranslation(coupling, modulation(coupling))

"""
    coupling(translation)

Return the coupling collection of an [`I2OTranslation`](@ref), or a bare collection itself.
"""
coupling(t::I2OTranslation) = t.coupling

"""
    modulation(translation)

Return the modulation policy of an [`I2OTranslation`](@ref). Bare coupling collections default
to [`NoModulation`](@ref); Toeplitz collections use [`ToeplitzModulation`](@ref).
"""
modulation(t::I2OTranslation) = t.modulation

# Node-pair keys

"""
    couplingkey(receivingnode, translatingnode, direction)

Return `(receivingnode, translatingnode)` for forward translation and the swapped pair for
transpose or adjoint translation. This selects the stored coupling; the kernel applies the
required transpose or conjugation separately.
"""
couplingkey(receivingnode::Int, translatingnode::Int, ::ForwardTranslation) =
    (receivingnode, translatingnode)
couplingkey(receivingnode::Int, translatingnode::Int, ::TransposeTranslation) =
    (translatingnode, receivingnode)
couplingkey(receivingnode::Int, translatingnode::Int, ::AdjointTranslation) =
    (translatingnode, receivingnode)

# Coupling application

# `applycoupling!` checks symmetry; concrete collections extend `applycouplingkernel!`.

"""
    applyi2o!(out, in, translation, receivingnode, translatingnode, direction, accumulation, context)

Apply the coupling for one node pair through [`applycoupling!`](@ref). Accepts an
[`I2OTranslation`](@ref) or a bare [`AbstractI2OCouplingCollection`](@ref).

Basis conversion is separate: call [`modulate_in!`](@ref) before the per-node I2O phase and
[`modulate_out!`](@ref) afterwards.
"""
function applyi2o!(
    out,
    in,
    translation::I2OTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::AbstractTranslationDirection,
    accumulation::AbstractAccumulationMode,
    context::TranslationContext,
)
    return applycoupling!(
        out,
        in,
        coupling(translation),
        receivingnode,
        translatingnode,
        direction,
        accumulation,
        context,
    )
end

# Bare coupling collections use the same guarded entry.
function applyi2o!(
    out,
    in,
    coupling::AbstractI2OCouplingCollection,
    receivingnode::Int,
    translatingnode::Int,
    direction::AbstractTranslationDirection,
    accumulation::AbstractAccumulationMode,
    context::TranslationContext,
)
    return applycoupling!(
        out, in, coupling, receivingnode, translatingnode, direction, accumulation, context
    )
end

# Per-node basis conversion

"""
    modulate_in!(moment, level, maptrait, modulation, coupling, channel, scheduler, buffer)

Convert `moment` into the coupling basis in place, before applying the node's I2O couplings.
Policies may use `coupling`, `channel`, and the borrowed `buffer` for plans or scratch.
[`NoModulation`](@ref) leaves it unchanged. A Toeplitz moment paired with `NoModulation` rejects
instead (see below) -- a Toeplitz moment always needs converting into the frequency domain, so
skipping that step must never happen silently, through ANY coupling collection.
"""
modulate_in!(
    moment, level::Int, maptrait, ::NoModulation, coupling, channel, scheduler, buffer
) = moment

"""
    modulate_out!(moment, level, maptrait, modulation, coupling, channel, scheduler, buffer)

Convert `moment` from the coupling basis back to its normal basis in place, after applying the
node's I2O couplings.
"""
modulate_out!(
    moment, level::Int, maptrait, ::NoModulation, coupling, channel, scheduler, buffer
) = moment

# A SECOND rejection boundary, additional to the coupling-typed one in `toeplitzi2ocollection.jl`
# (which still exists -- this does not replace it). This one dispatches on the MOMENT's type
# instead: a coupling collection that needs Toeplitz modulation can be reached through an
# arbitrary number of wrapper types (test mocks, future accelerated routes, ...), and a wrapper
# that does not forward `modulation` would otherwise fall through to the generic `NoModulation`
# no-op above with no indication the frequency-domain conversion was skipped -- exactly the
# failure this method exists to make loud instead of silent. Dispatching on
# `moment::AbstractToeplitzMoment` closes that gap for every such wrapper, present or future,
# without requiring each one to remember a delegation. See `toeplitzi2ocollection.jl` for the
# intersection methods resolving the ambiguity where both rejections would otherwise apply.
function modulate_in!(
    moment::AbstractToeplitzMoment,
    level::Int,
    maptrait,
    ::NoModulation,
    coupling,
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
    coupling,
    channel,
    scheduler,
    buffer,
)
    return throw(ArgumentError("Toeplitz couplings require ToeplitzModulation()"))
end

function modulate_in!(
    moment::AbstractToeplitzMoment,
    level::Int,
    maptrait,
    ::ToeplitzModulation,
    coupling,
    channel,
    scheduler,
    buffer,
)
    return _toeplitzmodulate_in!(moment, buffer)
end
function modulate_out!(
    moment::AbstractToeplitzMoment,
    level::Int,
    maptrait,
    ::ToeplitzModulation,
    coupling,
    channel,
    scheduler,
    buffer,
)
    return _toeplitzmodulate_out!(moment, buffer)
end

# Bare collections default to no basis conversion; specialized collections may override it.
coupling(x::AbstractI2OCouplingCollection) = x
modulation(x::AbstractI2OCouplingCollection) = NoModulation()

# Workspace scratch

"""
    i2ochannels(translation, translatingplan)

Build fresh I2O scratch channels for a workspace. Borrow buffers with
`borrowbuffer(channel, level, chunkid, maptrait)`.

An [`I2OTranslation`](@ref) delegates to its coupling collection. Collections needing no scratch
return [`EmptyChannels`](@ref); others provide their own method for buffers or FFT plans.
Each workspace must have its own scratch.

Call inside [`withschedulechunks`](@ref), since buffer counts use `nchunks(level)`.
"""
i2ochannels(t::I2OTranslation, translatingplan) = i2ochannels(coupling(t), translatingplan)
i2ochannels(::AbstractI2OCouplingCollection, translatingplan) = EmptyChannels()
