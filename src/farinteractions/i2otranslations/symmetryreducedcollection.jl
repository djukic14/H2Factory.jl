# Share canonical couplings using sample permutations and, where needed, component signs.

"""
    PermutedTranslation(canonical, permutation)

The translation `canonical` read through `permutation`: `t[j] == canonical[permutation[j]]`.

This lazy `AbstractArray` view lets one stored translation serve every interaction in its orbit.
The permutation uses the linear sample order.
"""
struct PermutedTranslation{T,N,A<:AbstractArray{T,N},P<:AbstractVector{<:Integer}} <:
       AbstractArray{T,N}
    canonical::A
    permutation::P

    function PermutedTranslation(
        canonical::A, permutation::P
    ) where {T,N,A<:AbstractArray{T,N},P<:AbstractVector{<:Integer}}
        length(canonical) == length(permutation) || throw(
            DimensionMismatch(
                "PermutedTranslation: the translation has $(length(canonical)) samples but the " *
                "permutation has $(length(permutation)). This is a level mismatch, the " *
                "permutation was built for a different level's angular grid.",
            ),
        )
        return new{T,N,A,P}(canonical, permutation)
    end
end

Base.size(translation::PermutedTranslation) = size(translation.canonical)
Base.IndexStyle(::Type{<:PermutedTranslation}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(translation::PermutedTranslation, j::Int)
    return translation.canonical[translation.permutation[j]]
end

"""
    ScaledPermutedTranslation(canonical, permutation, scale)

[`PermutedTranslation`](@ref) with a scalar factor: `t[j] == scale * canonical[permutation[j]]`.
The builder selects the field component; this view applies its sample permutation and scale.
"""
struct ScaledPermutedTranslation{
    T,N,A<:AbstractArray{<:Any,N},P<:AbstractVector{<:Integer},S
} <: AbstractArray{T,N}
    canonical::A
    permutation::P
    scale::S

    function ScaledPermutedTranslation(
        canonical::A, permutation::P, scale::S
    ) where {N,A<:AbstractArray{<:Any,N},P<:AbstractVector{<:Integer},S}
        length(canonical) == length(permutation) || throw(
            DimensionMismatch(
                "ScaledPermutedTranslation: the translation has $(length(canonical)) samples " *
                "but the permutation has $(length(permutation)). This is a level mismatch, " *
                "the permutation was built for a different level's grid.",
            ),
        )
        T = typeof(zero(S) * zero(eltype(A)))
        return new{T,N,A,P,S}(canonical, permutation, scale)
    end
end

Base.size(translation::ScaledPermutedTranslation) = size(translation.canonical)
Base.IndexStyle(::Type{<:ScaledPermutedTranslation}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(
    translation::ScaledPermutedTranslation, j::Int
)
    return translation.scale * translation.canonical[translation.permutation[j]]
end

"""
    SymmetryReducedI2OTranslationCollection

A coupling collection whose entries share canonical translations through permutations and,
when needed, component scales. Its storage trait selects the application kernel.

Built by [`symmetryreducedcollection`](@ref), [`symmetryreducedtoeplitzcollection`](@ref), or
[`symmetryreduceddensecollection`](@ref). Entries already account for their symmetries, so the
collection carries no unhandled tags and passes [`assertnosymmetry`](@ref).
"""
struct SymmetryReducedI2OTranslationCollection{DictType,I2OTrait} <:
       AbstractI2OTranslationCollection
    dict::DictType
    i2otrait::I2OTrait
    stats::TranslationStats
end

Base.getindex(tfs::SymmetryReducedI2OTranslationCollection, key) = tfs.dict[key]
Base.keys(tfs::SymmetryReducedI2OTranslationCollection) = keys(tfs.dict)

i2otrait(tfs::SymmetryReducedI2OTranslationCollection) = tfs.i2otrait
translationstats(tfs::SymmetryReducedI2OTranslationCollection) = tfs.stats

# The symmetry tags were consumed when this collection was built.
translationsymmetrygroup(::SymmetryReducedI2OTranslationCollection) = nothing
translationsymmetry(::SymmetryReducedI2OTranslationCollection, ::Int, ::Int) = nothing

# Each value already accounts for its symmetry.
assertnosymmetry(::SymmetryReducedI2OTranslationCollection) = nothing

function translation(
    tfs::SymmetryReducedI2OTranslationCollection, receivingnode::Int, translatingnode::Int
)
    return tfs.dict[(receivingnode, translatingnode)]
end

"""
    symmetryreducedcollection(canonical, permutationfor)

Wrap each stored coupling in a [`PermutedTranslation`](@ref). `canonical` must carry symmetry
tags, usually from `H2Trees.SymmetryDirectionInvariancePerLevel`.

`permutationfor(receivingnode, symmetry)` returns the sample-index permutation for that pair.
The supplied symmetry maps the canonical direction to the actual direction. For angular samples,
the callback uses its inverse: `T(k, S*d) == T(inv(S)*k, d)`.

The coupling and symmetry are read from the same stored key, including for transpose and adjoint.
Both nodes are at the same level, so `receivingnode` identifies the required sample grid.
"""
function symmetryreducedcollection(canonical::I2OTranslationCollection, permutationfor)
    group = translationsymmetrygroup(canonical)
    group === nothing && error(
        "symmetryreducedcollection: this collection carries no symmetry tags, so there is " *
        "nothing to permute and wrapping it would silently claim a reduction that did not " *
        "happen. Build it with `H2Trees.SymmetryDirectionInvariancePerLevel`.",
    )

    dict = Dict(
        key => PermutedTranslation(
            canonical[key],
            permutationfor(key[1], translationsymmetry(canonical, key[1], key[2])),
        ) for key in keys(canonical)
    )

    return SymmetryReducedI2OTranslationCollection(
        dict, i2otrait(canonical), translationstats(canonical)
    )
end

# Toeplitz sample permutations
#
# For a symmetry-invariant scalar kernel, κ_d[m] = k(d + h*m) gives κ_{Sd}[m] = κ_d[S⁻¹m].
# Axis permutations and circular reversals commute with the FFT, so the same index permutation
# works on the transformed tensor. Component signs are handled separately.

"""
    toeplitzpermutation(toeplitzsize, symmetry) -> Vector{Int32}

Return the linear-index permutation for a circulant-embedded Toeplitz difference tensor.
`toeplitzsize` is the stored size (`2n-1` per axis), not the moment size. Difference `d` is stored
at `mod(d, L) + 1` on an axis of length `L`.

The permutation applies the inverse symmetry to difference indices, so for a symmetry-invariant
scalar kernel, `kernel(S*d)[j] == kernel(d)[permutation[j]]`. It also applies to the FFT of the
embedded tensor. Field-component transformations are handled separately.

Axes exchanged by the symmetry must have equal lengths; otherwise this raises an error.
"""
function toeplitzpermutation(
    toeplitzsize::NTuple{D,<:Integer}, symmetry::H2Trees.LatticeSymmetry{D}
) where {D}
    for i in 1:D
        toeplitzsize[i] == toeplitzsize[symmetry.permutation[i]] || error(
            "toeplitzpermutation: $(symmetry) maps axis $(symmetry.permutation[i]) (length " *
            "$(toeplitzsize[symmetry.permutation[i]])) onto axis $i (length " *
            "$(toeplitzsize[i])). A signed coordinate permutation cannot change an axis length, " *
            "so this element does not act on a tensor of size $(toeplitzsize) at all. A " *
            "degenerate tensor admits only the subgroup fixing its long axes, choose a policy " *
            "that reflects the tensor's actual shape.",
        )
    end

    # Recover signed differences, apply the inverse symmetry, then wrap back to array indices.
    inverse = H2Trees.inversesymmetry(symmetry)
    linear = LinearIndices(toeplitzsize)
    permutation = Vector{Int32}(undef, prod(toeplitzsize))

    for index in CartesianIndices(toeplitzsize)
        difference = ntuple(D) do i
            L = toeplitzsize[i]
            raw = index[i] - 1
            return raw > L ÷ 2 ? raw - L : raw          # wrap-around order -> signed difference
        end
        image = H2Trees.applysymmetry(inverse, difference)
        source = ntuple(D) do i
            return mod(image[i], toeplitzsize[i]) + 1   # signed difference -> wrap-around order
        end
        permutation[linear[index]] = linear[CartesianIndex(source)]
    end
    return permutation
end

"""
    latticegridpermutation(gridsize, symmetry) -> Vector{Int32}

Return the linear-index permutation of a tensor grid: `permutation[i]` indexes the image of
point `i` under `symmetry`.

A sign flip reverses an axis as `i -> n + 1 - i`, unlike the circular reversal used by
[`toeplitzpermutation`](@ref). The grid points must be symmetric about the box centre; callers
verify this. Axes exchanged by the symmetry must have equal lengths, which is checked here.
"""
function latticegridpermutation(
    gridsize::NTuple{D,<:Integer}, symmetry::H2Trees.LatticeSymmetry{D}
) where {D}
    for i in 1:D
        gridsize[i] == gridsize[symmetry.permutation[i]] || throw(
            ArgumentError(
                "latticegridpermutation: $(symmetry) maps axis $(symmetry.permutation[i]) " *
                "(length $(gridsize[symmetry.permutation[i]])) onto axis $i (length " *
                "$(gridsize[i])). A signed coordinate permutation cannot change an axis length, " *
                "so this element does not act on a grid of size $(gridsize) at all.",
            ),
        )
    end

    linear = LinearIndices(gridsize)
    permutation = Vector{Int32}(undef, prod(gridsize))
    for index in CartesianIndices(gridsize)
        source = Tuple(index)
        image = ntuple(D) do i
            coordinate = source[symmetry.permutation[i]]
            # Reverse about the grid centre, without circular wrapping.
            return symmetry.signs[i] > 0 ? coordinate : gridsize[i] + 1 - coordinate
        end
        permutation[linear[index]] = linear[CartesianIndex(image)]
    end
    return permutation
end

# Field-component transformations
#
# Sample permutations do not account for field components. Vector and tensor rules also select
# source components and their signs; dropping those signs gives the wrong reflection action.

"""
    componentsource(symmetry, i) -> (source, scale)
    componentsource(symmetry, i, j) -> (sourcei, sourcej, scale)

Return the canonical component indices and scale for a 3-D vector or tensor transformed by
`symmetry`. For a signed coordinate permutation `S`:

    (S*v)[i] = signs[i] * v[perm[i]]
    (S*A*transpose(S))[i,j] = signs[i] * signs[j] * A[perm[i],perm[j]]

Apply the sample-index permutation separately.
"""
function componentsource(symmetry::H2Trees.LatticeSymmetry{3}, i::Int)
    return Int(symmetry.permutation[i]), Int(symmetry.signs[i])
end

function componentsource(symmetry::H2Trees.LatticeSymmetry{3}, i::Int, j::Int)
    return Int(symmetry.permutation[i]),
    Int(symmetry.permutation[j]),
    Int(symmetry.signs[i]) * Int(symmetry.signs[j])
end

"""
    VectorField()

Declare the 3-component polar-vector rule `v(S*x) == S*v(x)`, as for a scalar kernel's gradient.
"""
struct VectorField end

"""
    TrueTensor()

Declare the 3x3 tensor rule `G(S*x) == S*G(x)*transpose(S)`.
The kernel's transformation rule, not its stored shape, determines this choice.
"""
struct TrueTensor end

"""
    PseudoTensor()

Declare the 3x3 rule `K(S*x) == det(S)*S*K(x)*transpose(S)`.
Unlike [`TrueTensor`](@ref), this includes an extra sign under reflection.
"""
struct PseudoTensor end

"""
    componentsource(kind, symmetry, i, j) -> (sourcei, sourcej, scale)

[`componentsource`](@ref) with the parity rule applied: `kind` is [`TrueTensor`](@ref) or
[`PseudoTensor`](@ref).
"""
componentsource(::TrueTensor, symmetry::H2Trees.LatticeSymmetry{3}, i::Int, j::Int) =
    componentsource(symmetry, i, j)

function componentsource(
    ::PseudoTensor, symmetry::H2Trees.LatticeSymmetry{3}, i::Int, j::Int
)
    sourcei, sourcej, scale = componentsource(symmetry, i, j)
    return sourcei, sourcej, scale * H2Trees.determinant(symmetry)
end

# Build the reduced Toeplitz store
#
# Pass the tag unchanged: `toeplitzpermutation` already applies its inverse to sample indices.
# The caller declares the component action.

_reducedtranslation(value::ToeplitzMatrixRepresentation, permutation, symmetry, ::Nothing) =
    ToeplitzMatrixRepresentation(PermutedTranslation(value.toeplitz, permutation))

function _reducedtranslation(value::AbstractVector, permutation, symmetry, ::VectorField)
    return map(1:3) do i
        source, scale = componentsource(symmetry, i)
        return ToeplitzMatrixRepresentation(
            ScaledPermutedTranslation(value[source].toeplitz, permutation, scale)
        )
    end
end

function _reducedtranslation(
    value::AbstractMatrix, permutation, symmetry, kind::Union{TrueTensor,PseudoTensor}
)
    return [
        begin
            sourcei, sourcej, scale = componentsource(kind, symmetry, i, j)
            ToeplitzMatrixRepresentation(
                ScaledPermutedTranslation(
                    value[sourcei, sourcej].toeplitz, permutation, scale
                ),
            )
        end for i in 1:3, j in 1:3
    ]
end

# Reject mismatched storage and component rules rather than ignoring component signs.
function _reducedtranslation(value, permutation, symmetry, componentaction)
    return error(
        "symmetryreducedtoeplitzcollection: a coupling stored as $(typeof(value)) does not match " *
        "the declared component action $(componentaction). Expected a " *
        "`ToeplitzMatrixRepresentation` with `nothing`, a 3-vector with `VectorField()`, or a 3x3 " *
        "matrix with `TrueTensor()`/`PseudoTensor()`.",
    )
end

_toeplitztensorsize(value::ToeplitzMatrixRepresentation) = size(value.toeplitz)
_toeplitztensorsize(value::AbstractArray) = size(first(value).toeplitz)

"""
    symmetryreducedtoeplitzcollection(canonical; componentaction=nothing)

Share each symmetry orbit's stored Toeplitz tensor through [`toeplitzpermutation`](@ref).
`canonical` must carry symmetry tags, usually from `H2Trees.SymmetryDirectionInvariancePerLevel`.

Declare the kernel's component rule with `componentaction`: `nothing` for a scalar,
[`VectorField`](@ref) for a 3-vector, or [`TrueTensor`](@ref)/[`PseudoTensor`](@ref) for a 3x3 tensor.
The kernel must obey the declared symmetry rule; under that condition, reduction adds no
approximation.
"""
function symmetryreducedtoeplitzcollection(canonical; componentaction=nothing)
    translationsymmetrygroup(canonical) === nothing && error(
        "symmetryreducedtoeplitzcollection: this collection carries no symmetry tags, so there " *
        "is nothing to permute and wrapping it would silently claim a reduction that did not " *
        "happen. Build it with `H2Trees.SymmetryDirectionInvariancePerLevel`.",
    )

    # Share permutations by tensor size and symmetry. One vector per pair could erase the savings
    # from sharing coupling tensors.
    permutations = Dict{Tuple,Vector{Int32}}()

    dict = Dict(
        key => begin
            value = canonical[key]
            symmetry = translationsymmetry(canonical, key[1], key[2])
            # Keep untagged couplings unchanged.
            if symmetry === nothing
                value
            else
                toeplitzsize = _toeplitztensorsize(value)
                permutation = get!(permutations, (toeplitzsize, symmetry)) do
                    return toeplitzpermutation(toeplitzsize, symmetry)
                end
                _reducedtranslation(value, permutation, symmetry, componentaction)
            end
        end for key in keys(canonical)
    )

    return SymmetryReducedI2OTranslationCollection(
        dict, i2otrait(canonical), translationstats(canonical)
    )
end
