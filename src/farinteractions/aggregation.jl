"""
    aggregate!(A, x)

Run the upward pass for `A`, building node moments from the input vector `x`.
"""
function aggregate!(A, x::AbstractVector)
    return aggregate!(A, x, NoDiagnostics())
end

function aggregate!(A, x::AbstractVector, diagnostics::D) where {D<:AbstractDiagnostics}
    return aggregate!(A, x, aggregationplan(A), diagnostics)
end

function aggregate!(
    A, x::AbstractVector, aggregationplan::AggregatePlan, diagnostics::D
) where {D<:AbstractDiagnostics}
    return aggregatetree!(
        A,
        aggregationplan,
        aggregationschedule(A),
        aggregationtree(A),
        aggregationmoment(A),
        aggregationmomentcollection(A),
        i2itranslatorfunctor(A),
        leafmul(A),
        x,
        diagnostics,
    )
end

function aggregate!(
    A, x::AbstractVector, aggregationplan::AggregateTranslatePlan, diagnostics::D
) where {D<:AbstractDiagnostics}
    return aggregatetranslatetree!(
        A,
        aggregationplan,
        aggregationschedule(A),
        aggregationtree(A),
        aggregationmoment(A),
        disaggregationmoment(A),
        aggregationmomentcollection(A),
        i2itranslatorfunctor(A),
        i2otranslatorfunctor(A),
        leafmul(A),
        x,
        diagnostics,
    )
end

"""
    aggregatetree!(A, plan, schedule, tree, moment, momentcollection,
                   i2itranslation!, leafmul!, x, diagnostics)

Run the scheduled upward pass without I2O translations.

Leaf moments are built from `x`; internal moments are built from their children.
"""
function aggregatetree!(
    A,
    aggregationplan::AggregatePlan,
    schedule::PlanSchedule,
    tree,
    moment,
    momentcollection,
    i2itranslation!,
    leafmul!,
    x::AbstractVector,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    # `schedule.nodechunks[i]` is the precomputed, reused vector-of-vectors of aggregation nodes for
    # `schedule.levels[i]` -- no per-MVP `collect`/`index_chunks`. Each chunk is already a `Vector`.
    #
    # Each level's parallel block is a separate function call (not an inline closure) so `level` /
    # `levelchunks` reach the timed block as ARGUMENTS -- capturing loop-assigned variables in the
    # `with_timing` closure would box them and allocate even on the `NoDiagnostics` path.
    for (level, levelchunks) in zip(schedule.levels, schedule.nodechunks)
        _aggregatelevel!(
            A,
            aggregationplan,
            tree,
            moment,
            momentcollection,
            i2itranslation!,
            leafmul!,
            x,
            level,
            levelchunks,
            diagnostics,
        )
    end
    return nothing
end

function _aggregatelevel!(
    A,
    aggregationplan,
    tree,
    moment,
    momentcollection,
    i2itranslation!,
    leafmul!,
    x,
    level,
    levelchunks,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    with_timing(diagnostics, :aggregate, :level; level=level, mode=diagnosticmode(A)) do
        @tasks for chunkid in 1:length(levelchunks)
            @set scheduler = outerscheduler(A, AggregateOp(), level)
            aggregatechunk!(
                A,
                aggregationplan,
                tree,
                levelchunks[chunkid],
                moment,
                momentcollection,
                i2itranslation!,
                leafmul!,
                x,
                level,
                # `level + 1`: level-transfer scratch is keyed by the TRANSFER's own (child) level,
                # and this loop is over PARENTS -- each node aggregates the children one level down.
                # The I2O channel below is keyed by the node's own level, so it is not offset.
                borrowbuffer(i2ichannel(A), level + 1, chunkid, maptrait(A)),
                borrowbuffer(i2ochannel(A), level, chunkid, maptrait(A)),
                # Also unoffset, and for the same reason as the I2O channel: the leaf multiply
                # happens AT a node of this level (the `isleaf` branch of `upwardpass!`), so the
                # scratch it stages through belongs to that leaf's own level.
                borrowbuffer(aggregationleafchannel(A), level, chunkid, maptrait(A)),
            )
        end
    end
    return nothing
end

"""
    aggregatetranslatetree!(A, plan, schedule, tree, aggregationmoment,
                            disaggregationmoment, momentcollection,
                            i2itranslation!, translatei2o!, leafmul!, x,
                            diagnostics)

Run the scheduled upward pass and the I2O translations for each level.
"""
function aggregatetranslatetree!(
    A,
    aggregationplan::AggregateTranslatePlan,
    schedule::PlanSchedule,
    tree,
    aggregationmoment,
    disaggregationmoment,
    momentcollection,
    i2itranslation!,
    translatei2o!,
    leafmul!,
    x::AbstractVector,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    # Both phases consume precomputed, reused chunk vectors from the schedule: `nodechunks` for the
    # I2I upward pass, `receivingchunks` for the fused I2O phase. The two phases are barrier-separated
    # and timed as distinct level-phases (`:aggregate` then `:i2o`).
    for (level, levelchunks, levelreceiving) in
        zip(schedule.levels, schedule.nodechunks, schedule.receivingchunks)
        _aggregatelevel!(
            A,
            aggregationplan,
            tree,
            aggregationmoment,
            momentcollection,
            i2itranslation!,
            leafmul!,
            x,
            level,
            levelchunks,
            diagnostics,
        )
        _translatelevel!(
            A,
            aggregationplan,
            tree,
            disaggregationmoment,
            aggregationmoment,
            translatei2o!,
            level,
            levelreceiving,
            diagnostics,
        )
    end
    return nothing
end

function _translatelevel!(
    A,
    aggregationplan,
    tree,
    disaggregationmoment,
    aggregationmoment,
    translatei2o!,
    level,
    levelreceiving,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    with_timing(diagnostics, :i2o, :level; level=level, mode=diagnosticmode(A)) do
        @tasks for chunkid in 1:length(levelreceiving)
            @set scheduler = outerscheduler(A, I2OOp(), level)
            translatechunk!(
                A,
                aggregationplan,
                tree,
                levelreceiving[chunkid],
                disaggregationmoment,
                aggregationmoment,
                translatei2o!,
                level,
                borrowbuffer(i2ochannel(A), level, chunkid, maptrait(A)),
            )
        end
    end
    return nothing
end

# The trailing `modulate_in!` in the upward pass is gated on a mode-specific predicate: `storenode`
# for the store-only `AggregatePlan`, `istranslatingnode` for the fused `AggregateTranslatePlan`.
# Dispatch on the plan type keeps `aggregatechunk!` shared between both aggregate traversals.
_aggregatemodulates(plan::AggregatePlan, node) = H2Trees.storenode(plan, node)
_aggregatemodulates(plan::AggregateTranslatePlan, node) =
    H2Trees.istranslatingnode(plan, node)

# Serial per-chunk work for the aggregation upward pass: run `upwardpass!` on each node in the chunk
# (I2I children -> parent), then the gated `modulate_in!`. The buffers -- I2I, I2O and leaf -- are
# already borrowed for this chunk by the scheduling loop; every node in the chunk shares them,
# sequentially.
"""
    aggregatechunk!(A, plan, tree, nodes, moment, momentcollection,
                    i2itranslation!, leafmul!, x, level,
                    i2ibuffer, i2obuffer, leafbuffer)

Build the moments for one serial chunk of nodes in the upward pass.

The supplied buffers belong to this chunk and may be reused for every node in it.
"""
function aggregatechunk!(
    A,
    aggregationplan,
    tree,
    chunknodes,
    moment,
    momentcollection,
    i2itranslation!,
    leafmul!,
    x,
    level,
    i2ibuffer,
    i2obuffer,
    leafbuffer,
)
    # LEAVES FIRST, AS A CHUNK, then the internal nodes. This is not a reordering of anything
    # observable: at one level a node is either a leaf or an internal node, and `i2ipass!` only ever
    # reads CHILD moments a level down -- already complete when this level starts. The trailing
    # `modulate_in!` stays in the node loop below, so it still runs after the moment it modulates
    # is written, for leaves and internal nodes alike.
    aggregateleafchunk!(
        A, tree, chunknodes, moment, momentcollection, leafmul!, x, level, leafbuffer
    )
    for node in chunknodes
        H2Trees.isleaf(tree, node) ||
            i2ipass!(A, tree, node, moment, i2itranslation!, level, i2ibuffer)
        _aggregatemodulates(aggregationplan, node) && modulate_in!(
            moment(node), level, A, innerscheduler(A, ModulationOp(), level), i2obuffer
        )
    end
    return nothing
end

# Serial per-chunk work for the fused I2O phase: translate every receiving node in the chunk from its
# translating partners (`translateintonode!`), reusing this chunk's I2O buffer.
"""
    translatechunk!(A, plan, tree, nodes, disaggregationmoment,
                    aggregationmoment, translatei2o!, level, i2obuffer)

Apply all I2O translations received by one serial chunk of nodes.
"""
function translatechunk!(
    A,
    aggregationplan,
    tree,
    chunknodes,
    disaggregationmoment,
    aggregationmoment,
    translatei2o!,
    level,
    i2obuffer,
)
    for receivingnode in chunknodes
        translateintonode!(
            A,
            tree,
            receivingnode,
            H2Trees.translatingnodes(aggregationplan, receivingnode, level),
            disaggregationmoment,
            aggregationmoment,
            translatei2o!,
            level,
            innerscheduler(A, I2OOp(), level),
            i2obuffer,
        )
    end
    return nothing
end

# Map adapters: read the I2O policy and workspace channel from A, then call the protocol hooks.
"""
    modulate_in!(moment, level, A, scheduler, buffer)

Apply `A`'s I2O input conversion to `moment`. Read the policy and channel from `A`, then
delegate to the eight-argument modulation hook.
"""
function modulate_in!(moment, level::Int, A, scheduler::Scheduler, buffer)
    translation = i2otranslator(A)
    return modulate_in!(
        moment,
        level,
        maptrait(A),
        modulation(translation),
        coupling(translation),
        i2ochannel(A),
        scheduler,
        buffer,
    )
end

"""
    modulate_out!(moment, level, A, scheduler, buffer)

Apply `A`'s I2O output conversion to `moment`. Read the policy and channel from `A`, then
delegate to the eight-argument modulation hook.
"""
function modulate_out!(moment, level::Int, A, scheduler::Scheduler, buffer)
    translation = i2otranslator(A)
    return modulate_out!(
        moment,
        level,
        maptrait(A),
        modulation(translation),
        coupling(translation),
        i2ochannel(A),
        scheduler,
        buffer,
    )
end
