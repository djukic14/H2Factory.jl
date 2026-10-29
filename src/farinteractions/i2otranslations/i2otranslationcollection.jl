"""
    AbstractI2OTranslationCollection

Base type for coupling collections with node-pair lookup through `getindex`.
Basis conversion belongs to the modulation policy.
"""
abstract type AbstractI2OTranslationCollection <: AbstractI2OCouplingCollection end

# Construction
# Find translation IDs, choose one node pair per ID, build each translation, then index by pair.

struct TranslationMetadata{Infos,Directions}
    infos::Infos
    directions::Directions
    npairs::Int
    nunique::Int
end

function translationmetadata(tree, translatingplan, translationtrait)
    infos, directions = H2Trees.translations(tree, translatingplan, translationtrait)
    npairs = sum(length, infos; init=0)
    return TranslationMetadata(infos, directions, npairs, length(directions))
end

struct TranslationBuildPlan{R}
    representatives::R
    assigned::BitVector
end

# Pairs with the same translation ID share a coupling; keep the first as its representative.
function translationbuildplan(metadata::TranslationMetadata)
    representatives = Vector{NTuple{2,Int}}(undef, metadata.nunique)
    assigned = falses(metadata.nunique)
    for leveltfinfo in metadata.infos
        for tfinfo in leveltfinfo
            id = tfinfo.translationID
            if !assigned[id]
                representatives[id] = (tfinfo.receivingnode, tfinfo.translatingnode)
                assigned[id] = true
            end
        end
    end
    return TranslationBuildPlan(representatives, assigned)
end

# `direction` is the geometric translation direction, not a forward/transpose/adjoint mode.
translationeltype(builder) = eltype(builder)
function buildtranslation(builder, receivingnode::Int, translatingnode::Int, direction)
    return builder(receivingnode, translatingnode, direction)
end

function computetranslations(
    buildplan::TranslationBuildPlan, directions, builder; verbose=false
)
    uniquetranslations = Vector{translationeltype(builder)}(undef, length(directions))
    pbar = progressbar(length(directions) + 1, verbose; desc="Computing translations")
    # Skip IDs with no representative pair.
    @tasks for id in eachindex(directions)
        if buildplan.assigned[id]
            receivingnode, translatingnode = buildplan.representatives[id]
            uniquetranslations[id] = buildtranslation(
                builder, receivingnode, translatingnode, directions[id]
            )
            next!(pbar)
        end
    end
    next!(pbar; desc="Finished computing translations")
    finish!(pbar)
    return uniquetranslations
end

function pairtranslationstore(metadata::TranslationMetadata, uniquetranslations)
    translationpairs = Vector{NTuple{2,Int}}(undef, metadata.npairs)
    translations = Vector{eltype(uniquetranslations)}(undef, metadata.npairs)
    i = 0
    for leveltfinfo in metadata.infos
        for tfinfo in leveltfinfo
            i += 1
            translationpairs[i] = (tfinfo.receivingnode, tfinfo.translatingnode)
            translations[i] = uniquetranslations[tfinfo.translationID]
        end
    end
    return Dict(zip(translationpairs, translations))
end

# Construction statistics
# Keep reuse counts and build time for diagnostics.
struct TranslationStats
    npairs::Int
    nunique::Int
    pairsbylevel::Vector{Int}
    uniquebylevel::Vector{Int}
    constructionseconds::Float64
end

function _translationstats(metadata::TranslationMetadata, constructionseconds)
    pairsbylevel = Int[length(leveltfinfo) for leveltfinfo in metadata.infos]
    uniquebylevel = zeros(Int, length(metadata.infos))
    seen = falses(metadata.nunique)
    for (levelindex, leveltfinfo) in enumerate(metadata.infos)
        for tfinfo in leveltfinfo
            if !seen[tfinfo.translationID]
                seen[tfinfo.translationID] = true
                uniquebylevel[levelindex] += 1
            end
        end
    end
    return TranslationStats(
        metadata.npairs, metadata.nunique, pairsbylevel, uniquebylevel, constructionseconds
    )
end

reuseratio(s::TranslationStats) = iszero(s.nunique) ? 1.0 : s.npairs / s.nunique

# Symmetry metadata
# Store each pair's symmetry tag separately from its coupling. The tag maps the canonical
# direction to the pair's actual direction; its action on field components is handled elsewhere.
struct TranslationSymmetries{G}
    ids::Dict{NTuple{2,Int},UInt8}
    group::G
end

# The 2-D and 3-D lattice groups have 8 and 48 elements, so their IDs fit in UInt8.
function pairsymmetrystore(metadata::TranslationMetadata)
    ids = Dict{NTuple{2,Int},UInt8}()
    sizehint!(ids, metadata.npairs)
    for leveltfinfo in metadata.infos
        for tfinfo in leveltfinfo
            ids[(tfinfo.receivingnode, tfinfo.translatingnode)] = UInt8(tfinfo.symmetryID)
        end
    end
    return ids
end

function _hassymmetryids(metadata::TranslationMetadata)
    return :symmetryID in fieldnames(eltype(eltype(metadata.infos)))
end

# Collection
"""
    I2OTranslationCollection(dict, i2otrait)
    I2OTranslationCollection(tree, translatingplan, translationtrait, builder;
                             i2otranslationtrait=DiagonalTranslation(), verbose=false)
    I2OTranslationCollection(tree, aggregationplan, disaggregationplan, translationtrait,
                             builder; kwargs...)

Store I2O couplings under `(receivingnode, translatingnode)` keys.

The dictionary constructor returns a collection. The tree constructors compute each unique
translation once and return `(collection, EmptyChannels())`. The builder supplies `eltype(builder)`
and `builder(receivingnode, translatingnode, geometricdirection)`.

`translationtrait` controls reuse between node pairs; `i2otranslationtrait` selects how stored
couplings are applied. Symmetry-reduced construction retains tags that must be handled before
applying the couplings.
"""
struct I2OTranslationCollection{DictType,I2OTrait,S} <: AbstractI2OTranslationCollection
    dict::DictType
    i2otrait::I2OTrait
    stats::TranslationStats
    symmetry::S
end

# No symmetry table is needed for an unreduced collection.
function I2OTranslationCollection(dict, i2otrait, stats::TranslationStats)
    return I2OTranslationCollection(dict, i2otrait, stats, nothing)
end

function I2OTranslationCollection(
    tree, aggregationplan::A, disaggregationplan::D, translationtrait, builder; kwargs...
) where {A<:H2Trees.AbstractAggregationPlan,D<:H2Trees.AbstractDisaggregationPlan}
    translatingplan = H2Trees.translatingplan(aggregationplan, disaggregationplan)
    return I2OTranslationCollection(
        tree, translatingplan, translationtrait, builder; kwargs...
    )
end

function I2OTranslationCollection(
    tree,
    translatingplan,
    translationtrait,
    builder;
    i2otranslationtrait=DiagonalTranslation(),
    verbose=false,
)
    @assert H2Trees.istranslatingplan(translatingplan)

    metadata = translationmetadata(tree, translatingplan, translationtrait)
    buildplan = translationbuildplan(metadata)

    starttime = time_ns()
    uniquetranslations = computetranslations(
        buildplan, metadata.directions, builder; verbose=verbose
    )
    constructionseconds = (time_ns() - starttime) / 1e9

    store = pairtranslationstore(metadata, uniquetranslations)
    symmetry = if _hassymmetryids(metadata)
        TranslationSymmetries(
            pairsymmetrystore(metadata),
            H2Trees.symmetrygroup(translationtrait.policy, Val(length(eltype(tree)))),
        )
    else
        nothing
    end

    return I2OTranslationCollection(
        store,
        i2otranslationtrait,
        _translationstats(metadata, constructionseconds),
        symmetry,
    ),
    EmptyChannels()
end

# A prebuilt dictionary has no reuse metadata; count each entry as a unique translation.
function I2OTranslationCollection(dict, i2otrait)
    return I2OTranslationCollection(
        dict, i2otrait, TranslationStats(length(dict), length(dict), Int[], Int[], NaN)
    )
end

function translation(
    tfs::I2OTranslationCollection, receivingnode::Int, translatingnode::Int
)
    return tfs.dict[(receivingnode, translatingnode)]
end

function Base.getindex(tfs::I2OTranslationCollection, key)
    return tfs.dict[key]
end

"""
    keys(tfs::I2OTranslationCollection)

Every `(receivingnode, translatingnode)` key stored by the collection.

Derived collections should enumerate these keys instead of rebuilding the pair list independently.
"""
Base.keys(tfs::I2OTranslationCollection) = keys(tfs.dict)

function i2otrait(tfs::I2OTranslationCollection)
    return tfs.i2otrait
end

"""
    translationsymmetry(tfs, receivingnode, translatingnode)

Return the lattice symmetry mapping this pair's canonical direction to its actual direction,
or `nothing` when there are no symmetry tags.

To transform sample directions instead of the translation direction, use the inverse symmetry.
"""
function translationsymmetry(::I2OTranslationCollection{<:Any,<:Any,Nothing}, ::Int, ::Int)
    return nothing
end

function translationsymmetry(
    tfs::I2OTranslationCollection, receivingnode::Int, translatingnode::Int
)
    symmetry = tfs.symmetry
    return symmetry.group[symmetry.ids[(receivingnode, translatingnode)]]
end

"""
    translationsymmetrygroup(tfs)

The `H2Trees.SymmetryGroup` this collection's symmetry IDs index, or `nothing`.
"""
translationsymmetrygroup(::I2OTranslationCollection{<:Any,<:Any,Nothing}) = nothing
translationsymmetrygroup(tfs::I2OTranslationCollection) = tfs.symmetry.group

translationstats(tfs::I2OTranslationCollection) = tfs.stats

"""
    assertnosymmetry(tfs)

Reject collections with unhandled symmetry tags. Their stored canonical couplings cannot be
applied directly to every pair.

Collections must define a method: wrappers forward to their wrapped collection; types with no
unhandled tags return `nothing`. The default raises an error.
"""
assertnosymmetry(::I2OTranslationCollection{<:Any,<:Any,Nothing}) = nothing

function assertnosymmetry(tfs::AbstractI2OCouplingCollection)
    return error(
        "assertnosymmetry is not declared for $(typeof(tfs)), so the plain translation path " *
        "refuses it. Every collection reaching this path has to say which it is: forward to the " *
        "wrapped collection if this is a wrapper (`assertnosymmetry(tfs) = " *
        "assertnosymmetry(tfs.inner)`), or declare `assertnosymmetry(::T) = nothing` if it can " *
        "never carry unconsumed symmetry tags. There is no permissive default, because a wrapper " *
        "that silently inherited one would apply each orbit's canonical translation in place of " *
        "the interaction's own and return a plausible wrong field.",
    )
end

function assertnosymmetry(::I2OTranslationCollection)
    return error(
        "the plain translation path cannot apply a symmetry-reduced I2OTranslationCollection: " *
        "it stores one translation per symmetry orbit, so applying it directly would silently " *
        "use the canonical translation instead of this pair's own. Read `translationsymmetry(" *
        "tfs, receivingnode, translatingnode)` and apply the angular action in the consumer, " *
        "which is the only layer that knows what a symmetry does to its representation.",
    )
end

# Coupling application
# Keep the symmetry check in `applycoupling!`; collections specialize the kernel.

"""
    applycoupling!(out, in, coupling::AbstractI2OCouplingCollection, receivingnode,
                   translatingnode, direction, accumulation, context)

Check [`assertnosymmetry`](@ref), then apply the pair's coupling through
[`applycouplingkernel!`](@ref), using the collection's storage trait.

Implement the kernel, not this function: a more specific method here would bypass the check.
"""
function applycoupling!(
    out,
    in,
    translations::AbstractI2OCouplingCollection,
    receivingnode::Int,
    translatingnode::Int,
    direction::AbstractTranslationDirection,
    accumulation::AbstractAccumulationMode,
    context::TranslationContext,
)
    assertnosymmetry(translations)
    return applycouplingkernel!(
        out,
        in,
        translations,
        i2otrait(translations),
        receivingnode,
        translatingnode,
        direction,
        accumulation,
        context,
    )
end

# Direction

"""
    directedoperation(direction)

Return `identity`, `transpose`, or `adjoint` for the corresponding translation direction.
"""
directedoperation(::ForwardTranslation) = identity
directedoperation(::TransposeTranslation) = transpose
directedoperation(::AdjointTranslation) = adjoint

"""
    directedcoupling(block, direction)

Return `directedoperation(direction)(block)`. For ordinary matrices, transpose and adjoint
return lazy wrappers rather than copies.
"""
directedcoupling(block, direction::AbstractTranslationDirection) =
    directedoperation(direction)(block)

# Dense kernels
# The direction selects the node-pair key and the matrix operation.

function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::DenseTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::AbstractTranslationDirection,
    ::Overwrite,
    ::TranslationContext,
)
    LinearAlgebra.mul!(
        out,
        directedcoupling(
            tfs[couplingkey(receivingnode, translatingnode, direction)], direction
        ),
        in,
    )
    return out
end

function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::DenseTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::AbstractTranslationDirection,
    ::Accumulate,
    ::TranslationContext,
)
    LinearAlgebra.mul!(
        out,
        directedcoupling(
            tfs[couplingkey(receivingnode, translatingnode, direction)], direction
        ),
        in,
        true,
        true,
    )
    return out
end

# Diagonal kernels
# Forward and transpose use unchanged diagonal coefficients; adjoint conjugates them.
# Keep adjoint separate: the SVector overloads do not accept a `transform` keyword.

function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::DiagonalTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::Union{ForwardTranslation,TransposeTranslation},
    ::Overwrite,
    context::TranslationContext,
)
    return elementwisemul!(
        out,
        in,
        tfs[couplingkey(receivingnode, translatingnode, direction)];
        threading=context.scheduler,
    )
end

function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::DiagonalTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::Union{ForwardTranslation,TransposeTranslation},
    ::Accumulate,
    context::TranslationContext,
)
    return elementwisemul!(
        out,
        in,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        true,
        true;
        threading=context.scheduler,
    )
end

function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::DiagonalTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::AdjointTranslation,
    ::Overwrite,
    context::TranslationContext,
)
    return elementwisemul!(
        out,
        in,
        tfs[couplingkey(receivingnode, translatingnode, direction)];
        threading=context.scheduler,
        transform=conj,
    )
end

function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::DiagonalTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::AdjointTranslation,
    ::Accumulate,
    context::TranslationContext,
)
    return elementwisemul!(
        out,
        in,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        true,
        true;
        threading=context.scheduler,
        transform=conj,
    )
end
