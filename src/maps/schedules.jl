# Each plan has a chunk schedule built from its node lists. The schedule accessors select the same
# test or trial side as the plan accessors when a map is transposed or adjointed.

"""
    PlanSchedule(levels, nodechunks, receivingchunks, nchunks)

Hold the chunks for one plan. `levels[i]` identifies the level for `nodechunks[i]` and, when present,
`receivingchunks[i]`. Each chunk is a vector of node IDs.

`nchunks` is the requested count. A level with fewer nodes has fewer chunks; use
[`schedulechunkcounts`](@ref) to get the actual counts for buffer pools.
"""
struct PlanSchedule{C,R}
    levels::Vector{Int}
    nodechunks::C
    receivingchunks::R
    nchunks::Int
end

# Split one level's nodes into vectors that the traversal can reuse.
function _nodechunks(nodes::AbstractVector, nchunks::Int)
    return [nodes[c] for c in index_chunks(nodes; n=nchunks)]
end

function planschedule(plan::AggregatePlan, nchunks::Int)
    levels = collect(H2Trees.aggregationlevels(plan))
    nodechunks = [
        _nodechunks(collect(H2Trees.aggregationnodes(plan, level)), nchunks) for
        level in levels
    ]
    return PlanSchedule(levels, nodechunks, nothing, nchunks)
end

function planschedule(plan::AggregateTranslatePlan, nchunks::Int)
    levels = collect(H2Trees.aggregationlevels(plan))
    nodechunks = [
        _nodechunks(collect(H2Trees.aggregationnodes(plan, level)), nchunks) for
        level in levels
    ]
    receivingchunks = [
        _nodechunks(H2Trees.receivingnodes(plan, level), nchunks) for level in levels
    ]
    return PlanSchedule(levels, nodechunks, receivingchunks, nchunks)
end

function planschedule(plan::Union{DisaggregatePlan,DisaggregateTranslatePlan}, nchunks::Int)
    levels = collect(H2Trees.disaggregationlevels(plan))
    nodechunks = [
        _nodechunks(collect(H2Trees.disaggregationnodes(plan, level)), nchunks) for
        level in levels
    ]
    return PlanSchedule(levels, nodechunks, nothing, nchunks)
end

# Maps without stored schedules can build one from a plan when needed. This does not allocate buffers.
planschedule(plan) = planschedule(plan, Threads.nthreads())

"""
    Schedules(trialaggregation, testdisaggregation, testaggregation, trialdisaggregation)
    Schedules(plans; nchunks)

Hold one [`PlanSchedule`](@ref) for each test and trial aggregation and disaggregation plan.
`Schedules(plans; nchunks)` builds all four from the matching fields in `plans`.
"""
struct Schedules{TrA,TeD,TeA,TrD}
    trialaggregation::TrA
    testdisaggregation::TeD
    testaggregation::TeA
    trialdisaggregation::TrD
end

function Schedules(plans; nchunks::Int)
    return Schedules(
        planschedule(plans.trialaggregationplan, nchunks),
        planschedule(plans.testdisaggregationplan, nchunks),
        planschedule(plans.testaggregationplan, nchunks),
        planschedule(plans.trialdisaggregationplan, nchunks),
    )
end

"""
    schedulechunkcounts(schedule::PlanSchedule) -> Dict{Int,Int}
    schedulechunkcounts(schedules::Schedules) -> Dict{Int,Int}

Return the actual number of chunks needed at each level for scratch buffers.

For one plan, use the larger count from node chunks and receiving chunks. For a `Schedules` bundle,
use the largest count across its four plans at each level. This covers every map direction.
"""
function schedulechunkcounts(schedule::PlanSchedule)
    counts = Dict{Int,Int}()
    for (i, level) in enumerate(schedule.levels)
        n = length(schedule.nodechunks[i])
        if schedule.receivingchunks !== nothing
            n = max(n, length(schedule.receivingchunks[i]))
        end
        counts[level] = max(get(counts, level, 0), n)
    end
    return counts
end

function schedulechunkcounts(schedules::Schedules)
    counts = Dict{Int,Int}()
    for schedule in (
        schedules.trialaggregation,
        schedules.testdisaggregation,
        schedules.testaggregation,
        schedules.trialdisaggregation,
    )
        for (level, n) in schedulechunkcounts(schedule)
            counts[level] = max(get(counts, level, 0), n)
        end
    end
    return counts
end

"""
    withschedulechunks(f, schedules::Schedules)

Run `f()` with the actual per-level chunk counts available to `nchunks(level)`. Build buffer pools
inside this scope so they match the traversal's chunks.
"""
withschedulechunks(f, schedules::Schedules) = withnchunks(f, schedulechunkcounts(schedules))

"""
    checkexecutiondata(A)

Warn once if the schedule's requested chunk count is below the outer scheduler's worker count.
The comparison uses the map's own scheduler at the first aggregation level. Fewer chunks can leave
workers idle; it does not make buffer indexing unsafe. A smaller count may also be intentional to
save memory.
"""
function checkexecutiondata(A)
    schedule = aggregationschedule(A)
    built = schedule.nchunks
    isempty(schedule.levels) && return nothing
    wanted = requestedntasks(outerscheduler(A, AggregateOp(), first(schedule.levels)))
    if built < wanted
        @warn "H2 map chunk schedules/buffers were built for nchunks=$built but its outer scheduler asks for $wanted worker(s), so $(wanted - built) of them will sit idle. Rebuild with `nchunks >= $wanted` to use them -- or ignore this if the smaller partition is intentional (e.g. to cap buffer memory)." maxlog =
            1
    end
    return nothing
end
