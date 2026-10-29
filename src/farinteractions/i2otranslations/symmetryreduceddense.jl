# Keep the canonical matrix shared; gather input and scatter output around its multiplication:
#
#     w[j] = in[πc[j]]; y = M * w; out[πr[i]] = y[i]
#
# Transpose and adjoint swap the row and column permutations.

"""
    PermutedDenseTranslation(canonical, rowpermutation, columnpermutation)

A dense coupling represented by a shared canonical matrix and row/column permutations:
`t[rowpermutation[i], columnpermutation[j]] == canonical[i, j]`.

Application gathers the input, multiplies by the canonical matrix, then scatters the output.
This wrapper is not an `AbstractArray`; its dedicated kernel avoids copying the matrix.
"""
struct PermutedDenseTranslation{T,M<:AbstractMatrix{T},P<:AbstractVector{<:Integer}}
    canonical::M
    rowpermutation::P
    columnpermutation::P

    function PermutedDenseTranslation(
        canonical::M, rowpermutation::P, columnpermutation::P
    ) where {T,M<:AbstractMatrix{T},P<:AbstractVector{<:Integer}}
        length(rowpermutation) == size(canonical, 1) || throw(
            DimensionMismatch(
                "PermutedDenseTranslation: the canonical translation has $(size(canonical, 1)) " *
                "rows but the row permutation has $(length(rowpermutation)) entries. This is a " *
                "level mismatch, the permutation was built for a different level's grid.",
            ),
        )
        length(columnpermutation) == size(canonical, 2) || throw(
            DimensionMismatch(
                "PermutedDenseTranslation: the canonical translation has $(size(canonical, 2)) " *
                "columns but the column permutation has $(length(columnpermutation)) entries. " *
                "This is a level mismatch, the permutation was built for a different level's grid.",
            ),
        )
        return new{T,M,P}(canonical, rowpermutation, columnpermutation)
    end
end

# Use one grid permutation for both sides.
PermutedDenseTranslation(canonical::AbstractMatrix, permutation::AbstractVector) =
    PermutedDenseTranslation(canonical, permutation, permutation)

Base.eltype(::PermutedDenseTranslation{T}) where {T} = T
Base.size(translation::PermutedDenseTranslation) = size(translation.canonical)
Base.size(translation::PermutedDenseTranslation, i::Int) = size(translation.canonical, i)

"""
    SymmetryReducedDenseTranslation()

Trait for dense couplings applied through row/column permutations around a shared canonical
matrix. Supports [`PermutedDenseTranslation`](@ref) and [`PermutedDenseBlockTranslation`](@ref).
"""
struct SymmetryReducedDenseTranslation <: AbstractI2OTranslationTrait end

# Application kernels
#
# Borrow separate vectors for the gathered input and product. They must not alias `out`, since
# scattering could overwrite values still needed later.

@inline function _gather!(destination, source, permutation)
    @inbounds for j in eachindex(permutation)
        destination[j] = source[permutation[j]]
    end
    return destination
end

@inline function _scatter!(destination, source, permutation)
    @inbounds for i in eachindex(permutation)
        destination[permutation[i]] = source[i]
    end
    return destination
end

@inline function _scatteradd!(destination, source, permutation, α, β)
    @inbounds for i in eachindex(permutation)
        k = permutation[i]
        destination[k] = α * source[i] + β * destination[k]
    end
    return destination
end

# Forward gathers columns and scatters rows; transpose and adjoint reverse those roles.
_gatherpermutation(t::PermutedDenseTranslation, ::typeof(identity)) = t.columnpermutation
_scatterpermutation(t::PermutedDenseTranslation, ::typeof(identity)) = t.rowpermutation
_gatherpermutation(
    t::PermutedDenseTranslation, ::Union{typeof(transpose),typeof(adjoint)}
) = t.rowpermutation
_scatterpermutation(
    t::PermutedDenseTranslation, ::Union{typeof(transpose),typeof(adjoint)}
) = t.columnpermutation

# Validate in an inner constructor to avoid overwriting Julia's generated outer constructor.
"""
    PermutedDenseBlockTranslation(blocks, scales, rowpermutation, columnpermutation)

A multi-component dense coupling with shared canonical blocks. `blocks[a,b]` supplies output
component `a` from input component `b`, with factor `scales[a,b]` and the given grid permutations.

The builder selects the component blocks. Application passes each scale to `mul!`, so the
canonical matrices need not be copied or negated.
"""
struct PermutedDenseBlockTranslation{
    T,B<:AbstractMatrix{<:AbstractMatrix{T}},S,P<:AbstractVector{<:Integer}
}
    blocks::B
    scales::S
    rowpermutation::P
    columnpermutation::P

    function PermutedDenseBlockTranslation(
        blocks::B, scales::S, rowpermutation::P, columnpermutation::P
    ) where {T,B<:AbstractMatrix{<:AbstractMatrix{T}},S,P<:AbstractVector{<:Integer}}
        size(blocks) == size(scales) || throw(
            DimensionMismatch(
                "PermutedDenseBlockTranslation: $(size(blocks)) blocks but $(size(scales)) scales.",
            ),
        )
        for block in blocks
            length(rowpermutation) == size(block, 1) &&
            length(columnpermutation) == size(block, 2) || throw(
                DimensionMismatch(
                    "PermutedDenseBlockTranslation: a block is $(size(block)) but the " *
                    "permutations are $(length(rowpermutation))x$(length(columnpermutation)). " *
                    "This is a level mismatch, the permutation was built for a different " *
                    "level's grid.",
                ),
            )
        end
        return new{T,B,S,P}(blocks, scales, rowpermutation, columnpermutation)
    end
end

Base.eltype(::PermutedDenseBlockTranslation{T}) where {T} = T

_gatherpermutation(t::PermutedDenseBlockTranslation, ::typeof(identity)) =
    t.columnpermutation
_scatterpermutation(t::PermutedDenseBlockTranslation, ::typeof(identity)) = t.rowpermutation
_gatherpermutation(
    t::PermutedDenseBlockTranslation, ::Union{typeof(transpose),typeof(adjoint)}
) = t.rowpermutation
_scatterpermutation(
    t::PermutedDenseBlockTranslation, ::Union{typeof(transpose),typeof(adjoint)}
) = t.columnpermutation

# Transpose and adjoint swap block indices as well as transforming each matrix. Read the scale
# from that swapped block too; scales are real, so adjoint does not change them.
_blockindex(::typeof(identity), a, b) = (a, b)
_blockindex(::Union{typeof(transpose),typeof(adjoint)}, a, b) = (b, a)

# Sum the input blocks for one output component, then scatter. Reuse the same scratch pair.
function _permuteddenseblockmul!(
    out, in, translation::PermutedDenseBlockTranslation, scratch, operation, α, β
)
    gatherpermutation = _gatherpermutation(translation, operation)
    scatterpermutation = _scatterpermutation(translation, operation)
    gathered = view(scratch.gathered, eachindex(gatherpermutation))
    product = view(scratch.product, eachindex(scatterpermutation))

    for a in eachindex(out)
        overwrite = true
        for b in eachindex(in)
            ia, ib = _blockindex(operation, a, b)
            _gather!(gathered, wrapvector(in, b), gatherpermutation)
            LinearAlgebra.mul!(
                product,
                operation(translation.blocks[ia, ib]),
                gathered,
                translation.scales[ia, ib],
                overwrite ? false : true,
            )
            overwrite = false
        end
        _scatteradd!(wrapvector(out, a), product, scatterpermutation, α, β)
    end
    return out
end

_applypermuteddense!(
    out, in, translation::PermutedDenseBlockTranslation, scratch, operation
) = _permuteddenseblockmul!(out, in, translation, scratch, operation, true, false)

_applypermuteddense!(
    out, in, translation::PermutedDenseBlockTranslation, scratch, operation, α, β
) = _permuteddenseblockmul!(out, in, translation, scratch, operation, α, β)

# Apply one component; the moment-level methods below loop over components.
function _permuteddensemul!(
    outvector, invector, translation::PermutedDenseTranslation, scratch, operation
)
    gatherpermutation = _gatherpermutation(translation, operation)
    scatterpermutation = _scatterpermutation(translation, operation)
    gathered = _gather!(
        view(scratch.gathered, eachindex(gatherpermutation)), invector, gatherpermutation
    )
    product = LinearAlgebra.mul!(
        view(scratch.product, eachindex(scatterpermutation)),
        operation(translation.canonical),
        gathered,
    )
    return _scatter!(outvector, product, scatterpermutation)
end

function _permuteddensemul!(
    outvector, invector, translation::PermutedDenseTranslation, scratch, operation, α, β
)
    gatherpermutation = _gatherpermutation(translation, operation)
    scatterpermutation = _scatterpermutation(translation, operation)
    gathered = _gather!(
        view(scratch.gathered, eachindex(gatherpermutation)), invector, gatherpermutation
    )
    product = LinearAlgebra.mul!(
        view(scratch.product, eachindex(scatterpermutation)),
        operation(translation.canonical),
        gathered,
    )
    return _scatteradd!(outvector, product, scatterpermutation, α, β)
end

# Apply the same matrix to each moment component, reusing scratch between components.
function _applypermuteddense!(
    out, in, translation::PermutedDenseTranslation, scratch, operation
)
    for i in eachindex(out)
        _permuteddensemul!(
            wrapvector(out, i), wrapvector(in, i), translation, scratch, operation
        )
    end
    return out
end

function _applypermuteddense!(
    out, in, translation::PermutedDenseTranslation, scratch, operation, α, β
)
    for i in eachindex(out)
        _permuteddensemul!(
            wrapvector(out, i), wrapvector(in, i), translation, scratch, operation, α, β
        )
    end
    return out
end

# Untagged couplings use ordinary multiplication without scratch or permutations.
_applypermuteddense!(out, in, coupling::AbstractArray, scratch, operation) =
    LinearAlgebra.mul!(out, operation(coupling), in)

_applypermuteddense!(out, in, coupling::AbstractArray, scratch, operation, α, β) =
    LinearAlgebra.mul!(out, operation(coupling), in, α, β)

# The stored key selects the coupling and its permutations; the direction selects the operation.
function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::SymmetryReducedDenseTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::AbstractTranslationDirection,
    ::Overwrite,
    context::TranslationContext,
)
    return _applypermuteddense!(
        out,
        in,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        context.buffer,
        directedoperation(direction),
    )
end

function applycouplingkernel!(
    out,
    in,
    tfs::AbstractI2OTranslationCollection,
    ::SymmetryReducedDenseTranslation,
    receivingnode::Int,
    translatingnode::Int,
    direction::AbstractTranslationDirection,
    ::Accumulate,
    context::TranslationContext,
)
    return _applypermuteddense!(
        out,
        in,
        tfs[couplingkey(receivingnode, translatingnode, direction)],
        context.buffer,
        directedoperation(direction),
        true,
        true,
    )
end

# Build the reduced dense store
#
# Dense matrices do not retain the tensor-grid shape needed to build their permutations, so the
# caller supplies that shape.

_reduceddensetranslation(
    value::AbstractMatrix{<:Number}, rowpermutation, columnpermutation, symmetry, ::Nothing
) = PermutedDenseTranslation(value, rowpermutation, columnpermutation)

# Select the canonical component blocks and their signs at build time. The tensor rule includes
# the reflection sign where needed; application passes each scale to `mul!`.
function _reduceddensetranslation(
    value::AbstractMatrix{<:AbstractMatrix},
    rowpermutation,
    columnpermutation,
    symmetry,
    kind::Union{TrueTensor,PseudoTensor},
)
    sources = similar(value)
    scales = Matrix{Int8}(undef, size(value))
    for b in axes(value, 2), a in axes(value, 1)
        sourcea, sourceb, scale = componentsource(kind, symmetry, a, b)
        # Share the canonical matrix rather than copying it.
        sources[a, b] = value[sourcea, sourceb]
        scales[a, b] = scale
    end
    return PermutedDenseBlockTranslation(sources, scales, rowpermutation, columnpermutation)
end

# Reject mismatched storage and component rules. VectorField is not supported by this route.
function _reduceddensetranslation(
    value, rowpermutation, columnpermutation, symmetry, componentaction
)
    return error(
        "symmetryreduceddensecollection: a coupling stored as $(typeof(value)) does not match the " *
        "declared component action $(componentaction). Expected a matrix of numbers with " *
        "`nothing`, or a matrix of matrices with `TrueTensor()`/`PseudoTensor()`. `VectorField()` " *
        "is not supported on the dense route: the operators that declare it cannot form a Galerkin " *
        "map, and the Petrov dense route is refused separately.",
    )
end

"""
    symmetryreduceddensecollection(canonical, gridsizefor; componentaction=nothing)
    symmetryreduceddensecollection(canonical, rowgridsizefor, columngridsizefor; componentaction=nothing)

Share canonical dense couplings through row/column grid permutations. `canonical` must carry
symmetry tags, usually from `H2Trees.SymmetryDirectionInvariancePerLevel`.

`rowgridsizefor(node)` and `columngridsizefor(node)` return interpolation-grid shapes as tuples.
Rows belong to the receiving node, columns to the translating node. The two-argument form uses
one callback for both sides.

Use `componentaction=nothing` for scalar couplings, or [`TrueTensor`](@ref)/[`PseudoTensor`](@ref)
for 3x3 tensor couplings. [`VectorField`](@ref) is not supported.

The kernel must obey the declared symmetry rule and the grid points must be symmetric about
the box centre. Callers verify these properties; under those conditions, reduction adds no
approximation.
"""
symmetryreduceddensecollection(canonical, gridsizefor; componentaction=nothing) =
    symmetryreduceddensecollection(
        canonical, gridsizefor, gridsizefor; componentaction=componentaction
    )

function symmetryreduceddensecollection(
    canonical, rowgridsizefor, columngridsizefor; componentaction=nothing
)
    translationsymmetrygroup(canonical) === nothing && error(
        "symmetryreduceddensecollection: this collection carries no symmetry tags, so there is " *
        "nothing to permute and wrapping it would silently claim a reduction that did not " *
        "happen. Build it with `H2Trees.SymmetryDirectionInvariancePerLevel`.",
    )

    # Pairs with the same grid shape and symmetry can share a permutation.
    permutations = Dict{Tuple,Vector{Int32}}()
    permutationfor(gridsize, symmetry) = get!(permutations, (gridsize, symmetry)) do
        return latticegridpermutation(gridsize, symmetry)
    end

    dict = Dict(
        key => begin
            value = canonical[key]
            symmetry = translationsymmetry(canonical, key[1], key[2])
            # Keep untagged couplings unchanged; the apply methods also accept plain matrices.
            if symmetry === nothing
                value
            else
                _reduceddensetranslation(
                    value,
                    permutationfor(rowgridsizefor(key[1]), symmetry),
                    permutationfor(columngridsizefor(key[2]), symmetry),
                    symmetry,
                    componentaction,
                )
            end
        end for key in keys(canonical)
    )

    return SymmetryReducedI2OTranslationCollection(
        dict, SymmetryReducedDenseTranslation(), translationstats(canonical)
    )
end

# Use one representative block to size the level's scratch.
_canonicalmatrix(translation::PermutedDenseTranslation) = translation.canonical
_canonicalmatrix(translation::PermutedDenseBlockTranslation) = first(translation.blocks)
_canonicalmatrix(matrix::AbstractMatrix{<:Number}) = matrix
_canonicalmatrix(container::AbstractArray{<:AbstractArray}) = first(container)

# Scratch channel
#
# Each chunk needs its own gathered-input and product buffers.

struct PermutedDenseI2OChannel{S}
    scratch::S   # level => BufferStore of NamedTuples, one per chunk
end

function borrowbuffer(channel::PermutedDenseI2OChannel, level::Int, chunkid::Int, maptrait)
    !haskey(channel.scratch, level) && return EmptyChannels()
    return borrowbuffer(channel.scratch[level], chunkid)
end

# A reduced collection is not necessarily dense, so the trait selects its channels.
function i2ochannels(collection::SymmetryReducedI2OTranslationCollection, translatingplan)
    return _reducedi2ochannels(i2otrait(collection), collection, translatingplan)
end

_reducedi2ochannels(::AbstractI2OTranslationTrait, collection, translatingplan) =
    EmptyChannels()

function _reducedi2ochannels(::SymmetryReducedDenseTranslation, collection, translatingplan)
    scratchlevels = Int[]
    stores = []
    for level in H2Trees.levels(translatingplan)
        representative = _firstreducedtranslation(collection, translatingplan, level)
        representative === nothing && continue
        matrix = _canonicalmatrix(representative)
        # Size both buffers for the larger side, so they also fit transpose and adjoint.
        n = max(size(matrix, 1), size(matrix, 2))
        T = eltype(matrix)
        # Match the scheduled chunk count, not the number of threads.
        push!(
            stores,
            createchannel([
                (gathered=Vector{T}(undef, n), product=Vector{T}(undef, n)) for
                _ in 1:nchunks(level)
            ]),
        )
        push!(scratchlevels, level)
    end
    isempty(stores) && return EmptyChannels()
    return PermutedDenseI2OChannel(
        Dict(zip(scratchlevels, convert(Vector{typeof(stores[1])}, stores)))
    )
end

# Couplings at a level share a size; return the first for scratch sizing, or nothing if empty.
function _firstreducedtranslation(collection, translatingplan, level)
    for receivingnode in H2Trees.receivingnodes(translatingplan, level)
        for translatingnode in
            H2Trees.translatingnodes(translatingplan, receivingnode, level)
            return collection[(receivingnode, translatingnode)]
        end
    end
    return nothing
end
