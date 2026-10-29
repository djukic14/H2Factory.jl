"""
    AbstractStorer

Base type for callable objects that write assembled moment values. A storer returned by
[`buildstorer`](@ref) accepts `(value, cluster, m, n)`.
"""
abstract type AbstractStorer end

# `storetrait` and `buildstorer` are declared in `Interfaces.jl`. These are their defaults.
function storetrait(op)
    return storetrait(typeof(op))
end

function buildstorer(F, op, storeinfo)
    return storetrait(op)(F, storeinfo)
end

function buildstorer(Fs, op, storeinfo, tree)
    return storetrait(op)(Fs, storeinfo, tree)
end

"""
    StoreAsIs(Fs, storeinfo[, tree])

Add assembled values directly to `Fs`. With a tree, `StoreAsIs` maps global point indices to
their positions in each leaf; without one, it writes to a flat set of moment matrices.
"""
struct StoreAsIs{T,I} <: AbstractStorer
    Fs::T
    indexinleafbox::I
    function StoreAsIs(Fs::F, ::Any) where {F}
        return new{F,Nothing}(Fs, nothing)
    end
    function StoreAsIs(Fs::F, indexinleafbox::I) where {F,I<:AbstractArray}
        return new{F,I}(Fs, indexinleafbox)
    end
end

function StoreAsIs(Fs, ::Any, tree)
    return StoreAsIs(Fs, indexinleafbox(tree))
end

function (storer::StoreAsIs)(v, cluster, m, n)
    for i in eachindex(storer.Fs[cluster])
        storer.Fs[cluster][i][m, storer.indexinleafbox[cluster][n]] += v[i]
    end
    return nothing
end

# Without `indexinleafbox`, moments are flat; ignore the chunk id passed as `cluster`.
function (storer::StoreAsIs{T,Nothing})(v, cluster::Int, m, n) where {T}
    return storer(v, nothing, m, n)
end

function (storer::StoreAsIs)(v, ::Nothing, m, n)
    for i in eachindex(storer.Fs)
        storer.Fs[i][m, n] += v[i]
    end
    return nothing
end

function (storer::StoreAsIs{SVector{1,T}})(v, ::Nothing, m, n) where {T}
    storer.Fs[1][m, n] += v
    return nothing
end

function indexinleafbox(tree)
    leaves = H2Trees.leaves(tree)
    indexinleafbox = Vector{Dict{Int,Int}}(undef, length(leaves))

    for (i, leaf) in enumerate(leaves)
        points = H2Trees.values(tree, leaf)
        indexinleafbox[i] = Dict{Int,Int}(zip(points, 1:length(points)))
    end
    return indexinleafbox
end
