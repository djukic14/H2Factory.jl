
# Allocate scratch moments per level.
function storagemoments(relevantlevels, tree, moment)
    @assert !isempty(relevantlevels)

    storagemoments = Vector{Any}(undef, length(relevantlevels))
    for i in eachindex(relevantlevels)
        # One buffer per chunk at this level.
        storagemoments[i] = H2Factory.createchannel([
            moment(tree, relevantlevels[i]) for _ in 1:nchunks(relevantlevels[i])
        ])
    end
    storagemoments = convert(Vector{typeof(storagemoments[1])}, storagemoments)

    return storagemoments
end

"""
    storagemomentsplan(testaggregationplan, trialaggregationplan, tree, MomentType, momentsize)

Allocate node moments as views into pooled buffers. Return
`(buffers, testmoments, trialmoments)`.

# The aliasing contract

Moments share three memory pools:

  - `aggregationbuffer` holds stored nodes for the whole matvec.
  - `evenbuffer` and `oddbuffer` hold temporary moments, reused by levels of the same parity.

Temporary moments are live only across adjacent levels. Stored moments may be read later by I2O or
disaggregation, so they must not overlap other live moments.

[`checkmomentaliasing`](@ref) checks the assumptions that make reuse safe:

  - Aggregation levels are contiguous.
  - Nodes read after aggregation are stored, not temporary.
  - Nodes at the same level do not share a slice.

## Sharing Between Sides

Test and trial temporary moments may share memory. Their phases run one after the other, so they
are not live at the same time. Deferred I2O reads use stored moments. The aliasing check reports
cross-side temporary overlap as `:info`.

# Layout / allocation / materialization

[`planmomentstorage`](@ref) assigns slots to pools, [`allocatemomentbuffers`](@ref) allocates the
pools, and [`materializemoments`](@ref) wraps each slot. The plan can be inspected before allocation.

# Ownership

`buffers` keeps the wrapped memory alive. Keep it for as long as any moment view is in use; the
views do not keep the backing pool alive by themselves.
"""
function storagemomentsplan(
    testaggregationplan::AbstractAggregationPlan,
    trialaggregationplan::AbstractAggregationPlan,
    tree,
    MomentType,
    momentsize,
)
    return storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree,
        MomentType,
        MomentType,
        momentsize,
        momentsize,
        H2Trees.treetrait(tree),
    )
end

"""
    storagemomentsplan(testaggregationplan, trialaggregationplan, tree;
                       TestMomentType, TrialMomentType, testmomentsize, trialmomentsize)

Allocate test and trial moments with named type and size arguments. Use this form when the two
sides differ, since the positional form can silently swap their values.
"""
function storagemomentsplan(
    testaggregationplan::AbstractAggregationPlan,
    trialaggregationplan::AbstractAggregationPlan,
    tree;
    TestMomentType,
    TrialMomentType,
    testmomentsize,
    trialmomentsize,
)
    return storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree,
        TestMomentType,
        TrialMomentType,
        testmomentsize,
        trialmomentsize,
    )
end

function storagemomentsplan(
    testaggregationplan::AbstractAggregationPlan,
    trialaggregationplan::AbstractAggregationPlan,
    tree,
    TestMomentType,
    TrialMomentType,
    testmomentsize,
    trialmomentsize,
)
    return storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree,
        TestMomentType,
        TrialMomentType,
        testmomentsize,
        trialmomentsize,
        H2Trees.treetrait(tree),
    )
end

function storagemomentsplan(
    testaggregationplan::AbstractAggregationPlan,
    trialaggregationplan::AbstractAggregationPlan,
    tree,
    TestMomentType,
    TrialMomentType,
    testmomentsize,
    trialmomentsize,
    ::H2Trees.AbstractTreeTrait,
)
    return _storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree,
        tree,
        TestMomentType,
        TrialMomentType,
        testmomentsize,
        trialmomentsize,
    )
end

# Keep types before sizes: both this method and `_storagemomentsplan` use that order.
function storagemomentsplan(
    testaggregationplan::AbstractAggregationPlan,
    trialaggregationplan::AbstractAggregationPlan,
    tree,
    TestMomentType,
    TrialMomentType,
    testmomentsize,
    trialmomentsize,
    ::H2Trees.isBlockTree,
)
    return _storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        H2Trees.testtree(tree),
        H2Trees.trialtree(tree),
        TestMomentType,
        TrialMomentType,
        testmomentsize,
        trialmomentsize,
    )
end

function _storagemomentsplan(
    aggregatetranslateplan::AggregateTranslatePlan,
    aggregateplan::AggregatePlan,
    aggregatetranslatetree,
    aggregatetree,
    AggregateTranslateMomentType,
    AggregateMomentType,
    aggregatetranslatemomentsize,
    aggregatemomentsize,
)   # a: aggregate, at: aggregate translate
    buffers, amomentsdict, atmomentsdict = _storagemomentsplan(
        aggregateplan,
        aggregatetranslateplan,
        aggregatetree,
        aggregatetranslatetree,
        AggregateMomentType,
        AggregateTranslateMomentType,
        aggregatemomentsize,
        aggregatetranslatemomentsize,
    )
    return buffers, atmomentsdict, amomentsdict
end

function _storagemomentsplan(
    aggregateplan::AggregatePlan,
    aggregatetranslateplan::AggregateTranslatePlan,
    aggregatetree,
    aggregatetranslatetree,
    AggregateMomentType,
    AggregateTranslateMomentType,
    aggregatemomentsize,
    aggregatetranslatemomentsize,
)   # a: aggregate, at: aggregate translate
    # Three stages, split so the layout is a plain, inspectable description before any `unsafe_wrap`:
    #   planmomentstorage     -- assign each node to a pool (:aggregation/:even/:odd); no memory touched
    #   allocatemomentbuffers -- size and allocate the three pools
    #   materializemoments    -- wrap each planned slot into its pool
    # Both plans materialize into the SAME `buffers`: the aggregate plan owns the aggregation pool and
    # its own even/odd runs; the translate plan reuses the even/odd pools (each materialize resets to
    # the pool base per level), the deliberate, benign cross-side sharing `checkmomentaliasing` allows.
    aggregateslots = planmomentstorage(aggregateplan)
    aggregatetranslateslots = planmomentstorage(aggregatetranslateplan)

    buffers = allocatemomentbuffers(
        aggregateplan,
        aggregatetranslateplan,
        aggregatetree,
        aggregatetranslatetree,
        AggregateMomentType,
        AggregateTranslateMomentType,
        aggregatemomentsize,
        aggregatetranslatemomentsize,
    )

    amomentsdict = materializemoments(
        aggregateslots, aggregatetree, AggregateMomentType, aggregatemomentsize, buffers
    )
    atmomentsdict = materializemoments(
        aggregatetranslateslots,
        aggregatetranslatetree,
        AggregateTranslateMomentType,
        aggregatetranslatemomentsize,
        buffers,
    )
    return buffers, amomentsdict, atmomentsdict
end

function calculatememoryrequirements(
    aggregationplan::AggregatePlan, tree, MomentType, momentsize
)
    aggregationsize = 0
    evenbuffersize = 0
    oddbuffersize = 0

    for level in H2Trees.aggregationlevels(aggregationplan)
        numberofbuffermoments = 0
        numberofaggregatemoments = 0
        for node in H2Trees.aggregationnodes(aggregationplan, level)
            # if we store the node, we do not need to count it towards the buffer moments
            if H2Trees.storenode(aggregationplan, node)
                numberofaggregatemoments += 1
            else
                numberofbuffermoments += 1
            end
        end

        buffermoment = MomentType(level)(momentsize(tree, level))
        # `wrapbuffer` pads each array up to its alignment boundary, so the pool must budget that on
        # top of the entries themselves or aligning the last moment walks off the end. `alignmentslack`
        # is dispatched on the moment type: a Toeplitz moment wraps two arrays per component and aligns
        # the toeplitz one to FFTW's wider class, so it needs a bigger budget than a plain moment.
        numentries = numberofentries(buffermoment) + alignmentslack(buffermoment)

        aggregationsize += numberofaggregatemoments * numentries

        if iseven(level)
            evenbuffersize = max(evenbuffersize, numberofbuffermoments * numentries)
        else
            oddbuffersize = max(oddbuffersize, numberofbuffermoments * numentries)
        end
    end

    return aggregationsize, evenbuffersize, oddbuffersize
end

function calculatememoryrequirements(
    aggregationplan::AggregateTranslatePlan, tree, MomentType, momentsize
)
    evenbuffersize = 0
    oddbuffersize = 0

    for level in H2Trees.aggregationlevels(aggregationplan)
        numberofbuffermoments = length(H2Trees.aggregationnodes(aggregationplan, level))
        buffermoment = MomentType(level)(momentsize(tree, level))
        # See the `AggregatePlan` method above: the pool must budget `wrapbuffer`'s alignment padding
        # as well as the entries.
        numentries = numberofentries(buffermoment) + alignmentslack(buffermoment)

        if iseven(level)
            evenbuffersize = max(evenbuffersize, numberofbuffermoments * numentries)
        else
            oddbuffersize = max(oddbuffersize, numberofbuffermoments * numentries)
        end
    end

    return evenbuffersize, oddbuffersize
end

"""
    MomentSlot(node, level, pool)

One node's place in the pooled moment layout: which `pool` (`:aggregation`, `:even`, or `:odd`) its
moment lives in, at which tree `level`. Produced by [`planmomentstorage`](@ref) before any memory is
touched, consumed by [`materializemoments`](@ref) to do the `unsafe_wrap`. A plain description of the
layout that [`checkmomentaliasing`](@ref)-style checks can inspect without dereferencing anything.
"""
struct MomentSlot
    node::Int
    level::Int
    pool::Symbol
end

# Stored nodes use the persistent pool; other nodes use the pool for their level's parity.
_momentpool(storenode, node, level) =
    storenode(node) ? :aggregation : (iseven(level) ? :even : :odd)

"""
    planmomentstorage(plan[, storenode]) -> Vector{MomentSlot}

Describe an aggregation `plan`'s pooled moment layout as an ordered list of [`MomentSlot`](@ref)s --
which pool each node's moment belongs to -- WITHOUT allocating or wrapping anything. This is the
"plan" stage of `storagemomentsplan`: a plain value that can be inspected before any pointer exists.

Ordered by (level, node) exactly as [`materializemoments`](@ref) walks it, so the description and the
wrap it drives agree by construction. `storenode` defaults to the plan's own predicate -- an
`AggregatePlan` stores its translating nodes, an `AggregateTranslatePlan` stores nothing.
"""
planmomentstorage(plan::AggregatePlan) =
    planmomentstorage(plan, H2Trees.StoreNodeFunctor(plan))
function planmomentstorage(plan::AggregateTranslatePlan)
    return planmomentstorage(plan, H2Trees.StoreNoNodeFunctor(plan))
end

function planmomentstorage(plan, storenode)
    slots = MomentSlot[]
    for level in H2Trees.aggregationlevels(plan)
        for node in H2Trees.aggregationnodes(plan, level)
            push!(slots, MomentSlot(node, level, _momentpool(storenode, node, level)))
        end
    end
    return slots
end

"""
    allocatemomentbuffers(aggregateplan, aggregatetranslateplan,
                          aggregatetree, aggregatetranslatetree,
                          AggregateMomentType, AggregateTranslateMomentType,
                          aggregatemomentsize, aggregatetranslatemomentsize) -> NamedTuple

Allocate the three shared pools (`evenbuffer`, `oddbuffer`, `aggregationbuffer`), sized to hold both
plans' moments. Even/odd are sized to the larger of the two plans' per-parity needs (they reset to
the pool base per level and are reused across levels and across sides); the aggregation pool holds
every stored moment. Sizes come from [`calculatememoryrequirements`](@ref) -- the single source of
truth, so the pools cannot be sized inconsistently with what `materializemoments` wraps into them.

Each plan is sized against **its own** tree: for a `BlockTree`/Petrov map `aggregatetree` and
`aggregatetranslatetree` are the trial and test trees, and this is the same tree `materializemoments`
later wraps that plan's moments with -- so a size functor that actually depends on the tree cannot
under- or over-allocate one side. (Every current size functor ignores its tree argument, so this is a
robustness guarantee for a future one, not a change to any present result.)
"""
function allocatemomentbuffers(
    aggregateplan,
    aggregatetranslateplan,
    aggregatetree,
    aggregatetranslatetree,
    AggregateMomentType,
    AggregateTranslateMomentType,
    aggregatemomentsize,
    aggregatetranslatemomentsize,
)
    aggregatefirst = AggregateMomentType(H2Trees.levels(aggregatetree)[end])(
        aggregatemomentsize(aggregatetree, H2Trees.levels(aggregatetree)[end])
    )
    translatefirst = AggregateTranslateMomentType(
        H2Trees.levels(aggregatetranslatetree)[end]
    )(
        aggregatetranslatemomentsize(
            aggregatetranslatetree, H2Trees.levels(aggregatetranslatetree)[end]
        ),
    )
    # The three pools are one `MemoryBuffer{wrappertype}` each and back BOTH plans' moments, so the two
    # plans' moments must wrap the same scalar type. True for every current pair (all `ComplexF64`);
    # asserted so a mismatched user-defined moment pair fails here, loudly, instead of producing a pool
    # that is invalid for one side.
    wrappertype = eltype(eltype(aggregatefirst))
    @assert eltype(eltype(translatefirst)) === wrappertype "aggregate and translate moments must " *
        "wrap the same element type to share pooled buffers, got " *
        "$(eltype(eltype(aggregatefirst))) and $(eltype(eltype(translatefirst)))"

    atevenbuffersize, atoddbuffersize = calculatememoryrequirements(
        aggregatetranslateplan,
        aggregatetranslatetree,
        AggregateTranslateMomentType,
        aggregatetranslatemomentsize,
    )
    aggregationsize, aevenbuffersize, aoddbuffersize = calculatememoryrequirements(
        aggregateplan, aggregatetree, AggregateMomentType, aggregatemomentsize
    )

    evenbuffersize = max(aevenbuffersize, atevenbuffersize)
    oddbuffersize = max(aoddbuffersize, atoddbuffersize)

    MemoryBuffer = @memorybuffer
    return (
        evenbuffer=MemoryBuffer{wrappertype}(undef, evenbuffersize),
        oddbuffer=MemoryBuffer{wrappertype}(undef, oddbuffersize),
        aggregationbuffer=MemoryBuffer{wrappertype}(undef, aggregationsize),
    )
end

"""
    materializemoments(slots, tree, MomentType, momentsize, buffers) -> Dict{Int,<:AbstractMoment}

`unsafe_wrap` each planned [`MomentSlot`](@ref) into its pool, returning the `node => moment` dict.
The even/odd pointers reset to the pool base at each new level (transient scratch is reused level to
level); the aggregation pointer only ever advances (stored moments persist for the whole matvec).
`slots` is ordered by (level, node), so the per-level reset lands on level boundaries exactly as the
pool sizing in [`allocatemomentbuffers`](@ref) assumed.
"""
function materializemoments(slots, tree, MomentType, momentsize, buffers)
    evenbuffer, oddbuffer, aggregationbuffer = buffers

    evenpointer = pointer(evenbuffer)
    oddpointer = pointer(oddbuffer)
    aggregationpointer = pointer(aggregationbuffer)

    currentaggregationpointer = aggregationpointer
    currentevenpointer = evenpointer
    currentoddpointer = oddpointer
    currentlevel = nothing
    # The per-level reset only reuses the scratch correctly if each level's slots are contiguous;
    # revisiting a level after leaving it would reset its pool mid-fill and alias moments. That
    # grouping is `planmomentstorage`'s contract, but `materializemoments` is a reusable building
    # block, so enforce it rather than trust the caller.
    seenlevels = Set{Int}()

    nodes = Vector{Int}(undef, length(slots))
    moments = Vector{Any}(undef, length(slots))

    for (i, slot) in enumerate(slots)
        if slot.level != currentlevel
            slot.level in seenlevels && error(
                "materializemoments requires slots grouped by level; level $(slot.level) recurs " *
                "after another level -- the per-level pool reset would corrupt its moments",
            )
            # New level: the transient pools reset to base (same-parity levels reuse the scratch).
            push!(seenlevels, slot.level)
            currentevenpointer = evenpointer
            currentoddpointer = oddpointer
            currentlevel = slot.level
        end

        nodes[i] = slot.node
        momenttype = MomentType(slot.level)
        msize = momentsize(tree, slot.level)

        if slot.pool === :aggregation
            currentaggregationpointer, moment = wrapbuffer(
                aggregationpointer, currentaggregationpointer, momenttype, msize
            )
        elseif slot.pool === :even
            currentevenpointer, moment = wrapbuffer(
                evenpointer, currentevenpointer, momenttype, msize
            )
        elseif slot.pool === :odd
            currentoddpointer, moment = wrapbuffer(
                oddpointer, currentoddpointer, momenttype, msize
            )
        else
            error(
                "unknown moment pool $(repr(slot.pool)) for node $(slot.node); expected " *
                ":aggregation, :even, or :odd",
            )
        end
        moments[i] = moment
    end

    moments = convert(Vector{_getminmomenttype(moments)}, moments)
    return Dict(zip(nodes, moments))
end

# Build the union from unique moment types, keeping the splat small.
function _getminmomenttype(moments)
    return Union{unique(typeof.(moments))...}
end
