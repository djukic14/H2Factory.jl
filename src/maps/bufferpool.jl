# Scratch buffers are private to each chunk. A level may use fewer chunks than requested, so pools
# use the actual per-level counts from `Schedules`, passed through `withschedulechunks`.

const NCHUNKS = ScopedValue{Union{Nothing,AbstractDict{Int,Int}}}(nothing)

"""
    nchunks()

Always throws an error. Chunk counts vary by level; use `nchunks(level)` inside
[`withschedulechunks`](@ref).
"""
function nchunks()
    return error(
        "nchunks() has no meaning: chunk counts are per level. Call nchunks(level) inside a " *
        "`withschedulechunks(schedules) do ... end` block, so the pool matches that level's " *
        "actual chunk schedule.",
    )
end

"""
    nchunks(level::Int)

Return the number of chunks at `level` in the current schedule. Use inside
[`withschedulechunks`](@ref). A missing level raises `KeyError`.
"""
function nchunks(level::Int)
    counts = NCHUNKS[]
    isnothing(counts) && error(
        "nchunks($level) called outside a `withschedulechunks(schedules) do ... end` block: " *
        "there is no chunk schedule in scope to size the buffer pool from.",
    )
    return counts[level]
end

"""
    maxnchunks(levels::Int...)

Return the largest scheduled chunk count among `levels`, skipping missing levels.

Use this when a shared buffer pool serves traversals that borrow it using chunk IDs from different
levels. Throws an error if none of the levels has a schedule.
"""
function maxnchunks(levels::Int...)
    counts = NCHUNKS[]
    isnothing(counts) && error(
        "maxnchunks($(join(levels, ", "))) called outside a `withschedulechunks(schedules) do ... " *
        "end` block: there is no chunk schedule in scope to size the buffer pool from.",
    )
    best = 0
    for level in levels
        haskey(counts, level) && (best = max(best, counts[level]))
    end
    iszero(best) && error(
        "no scheduled chunk count for any of the levels $(levels): a buffer pool for levels the " *
        "traversal never visits is a sizing bug.",
    )
    return best
end

"""
    withnchunks(f, counts::AbstractDict{Int,Int})

Run `f()` with `counts` available to `nchunks(level)`. Prefer [`withschedulechunks`](@ref), which
gets the counts from `Schedules`.
"""
withnchunks(f, counts::AbstractDict{Int,Int}) = with(f, NCHUNKS => counts)
withnchunks(f, ::Nothing) = f()

"""
    BufferStore

Holds one scratch buffer per chunk. Use [`createchannel`](@ref) to build one.
"""
struct BufferStore{B}
    buffers::B   # one buffer per chunk
end

"""
    createchannel(buffers) -> BufferStore

Collect `buffers` into a chunk-indexed store. Provide a separate mutable buffer for each chunk;
this function does not copy the buffers themselves.
"""
function createchannel(buffer)
    return BufferStore(collect(buffer))
end

"""
    borrowbuffer(store::BufferStore, chunkid)
    borrowbuffer(channels, level, chunkid, trait)

Return the scratch buffer for a chunk. The level-indexed form returns [`EmptyChannels`](@ref) when
`channels` has no entry for `level`.
"""
function borrowbuffer(store::BufferStore, chunkid::Int)
    return store.buffers[chunkid]
end

"""
    EmptyChannels

Stand-in for an operation that needs no scratch buffer. Borrowing from it returns itself.
"""
struct EmptyChannels end

function borrowbuffer(e::EmptyChannels, ::Int)
    return e
end

function borrowbuffer(e::EmptyChannels, ::Int, ::Int, ::Any)
    return e
end

# A missing level has no scratch and returns `EmptyChannels`.
function borrowbuffer(
    channels::Union{AbstractArray,AbstractDict},
    level::Int,
    chunkid::Int,
    ::AbstractMapTrait,
)
    !haskey(channels, level) && return EmptyChannels()
    return borrowbuffer(channels[level], chunkid)
end
