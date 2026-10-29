# Level-transfer protocol for aggregation and disaggregation. A selector maps each tree edge to a
# stored transfer; `applytransfer!` applies its requested direction at the transfer boundary.

# --- Transfer-key selectors ---------------------------------------------------------------------

abstract type AbstractTransferSelector end

"""
    SectorSelector{D}

Map a tree edge to the child's sector in a `2^D` tree.

`D` is the sector-count exponent (`nsectors = 2^D`). `transferkey` verifies that the resulting key is
within `1:2^D`; [`leveltransfer`](@ref) checks it against the stored transfers.
"""
struct SectorSelector{D} <: AbstractTransferSelector
    # Validate the exponent before it reaches sector arithmetic.
    function SectorSelector{D}() where {D}
        (D isa Integer && D >= 0) || throw(
            ArgumentError(
                "SectorSelector{D}: D must be a nonnegative integer (the 2^D sector-count " *
                "exponent), got $(repr(D))::$(typeof(D))",
            ),
        )
        return new{D}()
    end
end

nsectors(::SectorSelector{D}) where {D} = 2^D

"""
    transferkey(selector, tree, parent, child)

Return the key for the `(parent, child)` edge in a translator's transfer storage.

`SectorSelector` converts the zero-based `H2Trees.sector` to a one-based key and bounds-checks it
against `1:2^D` unless called under `@inbounds`.
"""
function transferkey(::SectorSelector{D}, tree, parent::Int, child::Int) where {D}
    key = H2Trees.sector(tree, child) + 1
    @boundscheck checkbounds(Base.OneTo(2^D), key)
    return key
end

# --- Level translator ---------------------------------------------------------------------------

"""
    LevelTranslator(transfers, selector)

Store transfers and a selector mapping tree edges to keys.
`SectorSelector` uses `transfers[level][sector]`. The map accessor determines whether
the translator serves I2I or O2O.
"""
struct LevelTranslator{T,S<:AbstractTransferSelector}
    transfers::T
    selector::S
end

selector(t::LevelTranslator) = t.selector

"""
    leveltransfer(translator, tree, parent, child)

Resolve the stored transfer for the `(parent, child)` edge.

The default indexes by child level and selector key. Translators with another storage layout may
specialize this method.
"""
function leveltransfer(translator::LevelTranslator, tree, parent::Int, child::Int)
    key = transferkey(translator.selector, tree, parent, child)
    return translator.transfers[H2Trees.level(tree, child)][key]
end

# --- I2I translation collection -----------------------------------------------------------------
# Traits distinguish dense transfers from separable per-axis factors.

abstract type AbstractI2ITranslationTrait end

"""
    DenseI2I()

The stored transfer is one dense matrix applied by a single `mul!`.
"""
struct DenseI2I <: AbstractI2ITranslationTrait end

"""
    SeparableI2I()

The stored transfer contains per-axis matrices applied as a separable tensor contraction.
"""
struct SeparableI2I <: AbstractI2ITranslationTrait end

"""
    I2ITranslationCollection(transfers, selector, trait)

An I2I translator with sector-per-level storage, an edge selector, and a dense or separable
application trait. It implements [`leveltransfer`](@ref) and [`applytransfer!`](@ref).
"""
struct I2ITranslationCollection{
    S,Sel<:AbstractTransferSelector,Tr<:AbstractI2ITranslationTrait
}
    transfers::S
    selector::Sel
    trait::Tr
end

i2itrait(c::I2ITranslationCollection) = c.trait
selector(c::I2ITranslationCollection) = c.selector

# Present each stored transfer according to its application trait.
function leveltransfer(c::I2ITranslationCollection, tree, parent::Int, child::Int)
    raw = c.transfers[H2Trees.level(tree, child)][transferkey(
        c.selector, tree, parent, child
    )]
    return presenttransfer(c.trait, raw)
end
presenttransfer(::DenseI2I, raw) = raw
presenttransfer(::SeparableI2I, raw) = SeparableTransfer(raw)

# --- Transfer application -----------------------------------------------------------------------
# Direction selects the operator; transfer type selects how it is applied.
_directedtransfer(T, ::ForwardTranslation) = T
_directedtransfer(T, ::TransposeTranslation) = transpose(T)
_directedtransfer(T, ::AdjointTranslation) = adjoint(T)
_directedtransfer(T, ::ConjugateTranslation) = conj(T)

# Avoid the eager copy performed by `conj` for real arrays. Other real structured transfers should
# provide the same identity specialization when appropriate.
_directedtransfer(T::AbstractArray{<:Real}, ::ConjugateTranslation) = T

function applytransfer!(
    out,
    in,
    T,
    direction::AbstractTranslationDirection,
    ::Overwrite,
    context::TranslationContext,
)
    return applydirectedtransfer!(out, in, _directedtransfer(T, direction), false, context)
end
function applytransfer!(
    out,
    in,
    T,
    direction::AbstractTranslationDirection,
    ::Accumulate,
    context::TranslationContext,
)
    return applydirectedtransfer!(out, in, _directedtransfer(T, direction), true, context)
end

# Dense transfers use one `mul!`; structured transfers provide their own application method.
function applydirectedtransfer!(
    out, in, T::AbstractMatrix, accumulate::Bool, ::TranslationContext
)
    accumulate ? LinearAlgebra.mul!(out, T, in, true, true) : LinearAlgebra.mul!(out, T, in)
    return out
end

# --- Separable transfer --------------------------------------------------------------------------

"""
    SeparableTransfer(matrices)

Store per-axis matrices for a separable contraction, retaining the original references.
`transpose`, `adjoint`, and `conj` act on each factor.
"""
struct SeparableTransfer{T}
    matrices::T
end

Base.transpose(t::SeparableTransfer) = SeparableTransfer(map(transpose, t.matrices))
Base.adjoint(t::SeparableTransfer) = SeparableTransfer(map(adjoint, t.matrices))
# Conjugation distributes over the factors; real factors avoid an eager copy.
Base.conj(t::SeparableTransfer) = SeparableTransfer(map(_conjfactor, t.matrices))
_conjfactor(m) = conj(m)
_conjfactor(m::AbstractArray{<:Real}) = m

# Direction has already been applied to `transfer.matrices`. Scratch is shape-sensitive: aggregation
# and disaggregation channels are sized for their respective contraction orientations, including
# non-square per-axis matrices.
function applydirectedtransfer!(
    out, in, transfer::SeparableTransfer, accumulate::Bool, context::TranslationContext
)
    for i in eachindex(out)
        !accumulate && (out[i] .= 0)
        _applyonedimseparable!(out[i], in[i], transfer.matrices, context.buffer, true)
    end
    return out
end

"""
    _onedimbuffershapes(n_in::Int, n_out::Int, D::Int)

Return the shapes of the `D-1` intermediate buffers for a separable contraction.

After step `s`, the last `s` axes have size `n_out` and the remaining axes have size `n_in`. No
buffer is needed when `D == 1`.
"""
function _onedimbuffershapes(n_in::Int, n_out::Int, D::Int)
    return ntuple(D - 1) do s
        ntuple(D) do k
            return k <= D - s ? n_in : n_out
        end
    end
end

"""
    _applyonedimseparable!(output, input, resmatrices::NTuple{D}, buffers, α) where {D}

Apply the separable contraction and accumulate into `output`.
Reuse the `D-1` buffers from [`_onedimbuffershapes`](@ref).
Scale by `α` on the first step only. Axes run from `D` to `1`;
the generated function emits the nested loops at compile time.
"""
@generated function _applyonedimseparable!(
    output, input, resmatrices::NTuple{D}, buffers, α
) where {D}
    stmts = Expr[]
    for s in 1:D
        d = D - s + 1
        otherdims = [k for k in 1:D if k != d]
        idxsyms = [Symbol(:idx_, k) for k in otherdims]

        invar = s == 1 ? :input : :(buffers[$(s - 1)])
        outvar = s == D ? :output : :(buffers[$s])

        inidx = Vector{Any}(undef, D)
        outidx = Vector{Any}(undef, D)
        for (j, k) in enumerate(otherdims)
            inidx[k] = idxsyms[j]
            outidx[k] = idxsyms[j]
        end
        inidx[d] = :(:)
        outidx[d] = :(:)

        αexpr = s == 1 ? :α : true

        call = :(LinearAlgebra.mul!(
            view($outvar, $(outidx...)),
            resmatrices[$d],
            view($invar, $(inidx...)),
            $αexpr,
            true,
        ))

        stepbody = call
        for k in otherdims
            stepbody = :(
                for $(Symbol(:idx_, k)) in axes($outvar, $k)
                    $stepbody
                end
            )
        end
        push!(stmts, stepbody)
    end

    zerostmts = [:(buffers[$k] .= 0) for k in 1:(D - 1)]

    return quote
        $(zerostmts...)
        $(stmts...)
        return output
    end
end

# --- Generic entry point ------------------------------------------------------------------------

"""
    leveltranslation!(out, in, translator, tree, parent, child, direction, accumulation, context)

Apply the `(parent, child)` transfer in `direction`, overwriting or accumulating into `out`.
Resolve it with [`leveltransfer`](@ref), then apply it with [`applytransfer!`](@ref).
Any object implementing `leveltransfer` can serve as a translator; the wrapper is optional.
"""
function leveltranslation!(
    out,
    in,
    translator,
    tree,
    parent::Int,
    child::Int,
    direction::AbstractTranslationDirection,
    accumulation::AbstractAccumulationMode,
    context::TranslationContext,
)
    T = leveltransfer(translator, tree, parent, child)
    applytransfer!(out, in, T, direction, accumulation, context)
    return out
end

# --- Workspace-local scratch for a level-transfer collection ---------------------------------------
#
# Each workspace owns separable-contraction scratch sized directly from its stored transfers.

"""
    i2ichannels(collection::I2ITranslationCollection)

Return workspace-local aggregation scratch keyed by the transfer's child level.

Dense transfers return [`EmptyChannels`](@ref). Separable transfers allocate per-chunk intermediate
buffers and must be built inside [`withschedulechunks`](@ref).
"""
i2ichannels(c::I2ITranslationCollection) = i2ichannels(c, i2itrait(c))
i2ichannels(::I2ITranslationCollection, ::DenseI2I) = EmptyChannels()

function i2ichannels(c::I2ITranslationCollection, ::SeparableI2I)
    isnothing(c.transfers) && return EmptyChannels()
    # Stored factors map child moments to parent moments.
    return _separablechannels(
        c, m -> (size(m, 2), size(m, 1)), level -> maxnchunks(level - 1)
    )
end

"""
    o2ochannels(collection::I2ITranslationCollection)

Return workspace-local disaggregation scratch keyed by the transfer's child level.

Disaggregation applies the transpose or adjoint of the aggregation transfer, so intermediate shapes
are reversed. Its pool uses [`maxnchunks`](@ref) across the parent and child schedules because the two
downward passes draw chunk identifiers from different levels.
"""
o2ochannels(c::I2ITranslationCollection) = o2ochannels(c, i2itrait(c))
o2ochannels(::I2ITranslationCollection, ::DenseI2I) = EmptyChannels()

function o2ochannels(c::I2ITranslationCollection, ::SeparableI2I)
    isnothing(c.transfers) && return EmptyChannels()
    return _separablechannels(
        c, m -> (size(m, 1), size(m, 2)), level -> maxnchunks(level - 1, level)
    )
end

# One buffer set per transfer level and chunk. `shape` returns `(n_in, n_out)` for a representative
# factor; `poolsize` gives the number of chunk-local sets required at that level.
function _separablechannels(c::I2ITranslationCollection, shape, poolsize)
    channels = []
    keylevels = Int[]
    for (level, sectortransfers) in c.transfers
        factors = sectortransfers[1]
        D = length(factors)
        representative = factors[1]
        n_in, n_out = shape(representative)
        buffers = Tuple(
            zeros(complex(eltype(representative)), s...) for
            s in _onedimbuffershapes(n_in, n_out, D)
        )
        push!(channels, createchannel([deepcopy(buffers) for _ in 1:poolsize(level)]))
        push!(keylevels, level)
    end
    isempty(channels) && return EmptyChannels()
    return Dict(zip(keylevels, convert(Vector{typeof(channels[1])}, channels)))
end
