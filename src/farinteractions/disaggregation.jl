"""
    disaggregate!(y, A)
    disaggregate!(y, A, α, β)

Run the downward pass for `A` and scatter leaf moments into `y`. The four-argument form scales the
new contribution by `α` and the existing contents of `y` by `β`.
"""
function disaggregate!(y, A)
    return disaggregate!(y, A, NoDiagnostics())
end

function disaggregate!(y, A, diagnostics::D) where {D<:AbstractDiagnostics}
    return disaggregate!(y, A, disaggregationplan(A), diagnostics)
end

function disaggregate!(y, A, α::Number, β::Number)
    return disaggregate!(y, A, α, β, NoDiagnostics())
end

function disaggregate!(
    y, A, α::Number, β::Number, diagnostics::D
) where {D<:AbstractDiagnostics}
    return disaggregate!(y, A, disaggregationplan(A), α, β, diagnostics)
end

function disaggregate!(
    y, A, disaggregationplan::DisaggregatePlan, diagnostics::D
) where {D<:AbstractDiagnostics}
    return disaggregatetree!(
        A,
        disaggregationplan,
        disaggregationschedule(A),
        disaggregationtree(A),
        disaggregationmoment(A),
        disaggregationmomentcollection(A),
        o2otranslatorfunctor(A),
        y,
        leafmul(A),
        diagnostics,
    )
end

function disaggregate!(
    y, A, disaggregationplan::DisaggregatePlan, α, β, diagnostics::D
) where {D<:AbstractDiagnostics}
    return disaggregatetree!(
        A,
        disaggregationplan,
        disaggregationschedule(A),
        disaggregationtree(A),
        disaggregationmoment(A),
        disaggregationmomentcollection(A),
        o2otranslatorfunctor(A),
        y,
        ShiftedMul(leafmul(A), α, β),
        diagnostics,
    )
end

function disaggregate!(
    y, A, disaggregationplan::DisaggregateTranslatePlan, diagnostics::D
) where {D<:AbstractDiagnostics}
    return disaggregatetranslatetree!(
        A,
        disaggregationplan,
        disaggregationschedule(A),
        disaggregationtree(A),
        aggregationmoment(A),
        disaggregationmoment(A),
        disaggregationmomentcollection(A),
        i2otranslatorfunctor(A),
        o2otranslatorfunctor(A),
        y,
        leafmul(A),
        diagnostics,
    )
end

function disaggregate!(
    y, A, disaggregationplan::DisaggregateTranslatePlan, α, β, diagnostics::D
) where {D<:AbstractDiagnostics}
    return disaggregatetranslatetree!(
        A,
        disaggregationplan,
        disaggregationschedule(A),
        disaggregationtree(A),
        aggregationmoment(A),
        disaggregationmoment(A),
        disaggregationmomentcollection(A),
        i2otranslatorfunctor(A),
        o2otranslatorfunctor(A),
        y,
        ShiftedMul(leafmul(A), α, β),
        diagnostics,
    )
end

"""
    disaggregatetranslatetree!(A, plan, schedule, tree, aggregationmoment,
                               disaggregationmoment, momentcollection,
                               translatei2o!, o2otranslation!, y, leafmul!,
                               diagnostics)

Run the scheduled I2O and downward passes, then scatter leaf moments into `y`.
"""
function disaggregatetranslatetree!(
    A,
    disaggregationplan::DisaggregateTranslatePlan,
    schedule::PlanSchedule,
    tree,
    aggregationmoment,
    disaggregationmoment,
    momentcollection,
    translatei2o!,
    o2otranslation!,
    y::AbstractVector,
    leafmul!,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    rootnode = H2Trees.root(tree)
    # `schedule.nodechunks[i]` is the precomputed, reused vector-of-vectors of disaggregation nodes
    # for `schedule.levels[i]`. Each level's `@tasks` block is a separate function call so `level` /
    # `levelchunks` are arguments, not loop-captured closure variables (avoids boxing on the no-op
    # path). The fused level does I2O then O2O in one loop body, timed together as `:i2o_disaggregate`.
    for (level, levelchunks) in zip(schedule.levels, schedule.nodechunks)
        _disaggregatetranslatelevel!(
            A,
            disaggregationplan,
            tree,
            rootnode,
            aggregationmoment,
            disaggregationmoment,
            momentcollection,
            translatei2o!,
            o2otranslation!,
            y,
            leafmul!,
            level,
            levelchunks,
            diagnostics,
        )
    end
    return nothing
end

function _disaggregatetranslatelevel!(
    A,
    disaggregationplan,
    tree,
    rootnode,
    aggregationmoment,
    disaggregationmoment,
    momentcollection,
    translatei2o!,
    o2otranslation!,
    y,
    leafmul!,
    level,
    levelchunks,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    # NB: `:i2o_disaggregate` is FUSED timing -- this level's loop body does the I2O translate-in AND
    # the O2O push-down per node, so the record covers both, not pure I2O. (The store-mode
    # `disaggregatetree!` is O2O-only and records `:disaggregate`.)
    with_timing(
        diagnostics, :i2o_disaggregate, :level; level=level, mode=diagnosticmode(A)
    ) do
        @tasks for chunkid in 1:length(levelchunks)
            @set scheduler = outerscheduler(A, DisaggregateOp(), level)
            disaggregatetranslatechunk!(
                A,
                disaggregationplan,
                tree,
                rootnode,
                levelchunks[chunkid],
                aggregationmoment,
                disaggregationmoment,
                momentcollection,
                translatei2o!,
                o2otranslation!,
                y,
                leafmul!,
                level,
                borrowbuffer(i2ochannel(A), level, chunkid, maptrait(A)),
                borrowbuffer(o2ochannel(A), level, chunkid, maptrait(A)),
                # Leaf scratch is keyed by the LEAF's own level, and this loop is over the nodes of
                # `level` themselves (the leaf branch below fires when one of them is a leaf), so no
                # offset -- unlike the store-mode pass, which loops parents.
                borrowbuffer(disaggregationleafchannel(A), level, chunkid, maptrait(A)),
            )
        end
    end
    return nothing
end

# Serial per-chunk work for the fused disaggregate-translate downward pass. For each node in the
# chunk: translate in the far-field contribution (`translateintonode!`), then push the parent's
# moment down (O2O), then either recurse (internal node) or scatter to the leaf DOFs (`leafmul!`).
# The `initializedmoment` flag records whether I2O or O2O wrote the node.
"""
    disaggregatetranslatechunk!(A, plan, tree, root, nodes,
                                aggregationmoment, disaggregationmoment,
                                momentcollection, translatei2o!,
                                o2otranslation!, y, leafmul!, level,
                                i2obuffer, o2obuffer, leafbuffer)

Process one serial chunk of the fused I2O and downward pass.

Each node receives its I2O contribution before its parent contribution is pushed down. Initialized
leaf moments are then scattered into `y`.
"""
function disaggregatetranslatechunk!(
    A,
    disaggregationplan,
    tree,
    rootnode,
    chunknodes,
    aggregationmoment,
    disaggregationmoment,
    momentcollection,
    translatei2o!,
    o2otranslation!,
    y,
    leafmul!,
    level,
    i2obuffer,
    o2obuffer,
    leafbuffer,
)
    for node in chunknodes
        initializedmoment = translateintonode!(
            A,
            tree,
            node,
            H2Trees.translatingnodes(disaggregationplan, node),
            disaggregationmoment,
            aggregationmoment,
            translatei2o!,
            level,
            innerscheduler(A, I2OOp(), level),
            i2obuffer,
        )

        if node != rootnode
            parent = H2Trees.parent(tree, node)
            if H2Trees.isdisaggregationnode(disaggregationplan, parent)
                if initializedmoment
                    o2otranslation!(
                        disaggregationmoment(node),
                        disaggregationmoment(parent),
                        tree,
                        parent,
                        node,
                        Accumulate(),
                        innerscheduler(A, O2OOp(), level),
                        o2obuffer,
                    )
                else
                    o2otranslation!(
                        disaggregationmoment(node),
                        disaggregationmoment(parent),
                        tree,
                        parent,
                        node,
                        innerscheduler(A, O2OOp(), level),
                        o2obuffer,
                    )
                    initializedmoment = true
                end
            end
        end
    end

    # LEAF SCATTER AS A CHUNK, after every node's I2O and O2O -- a POST-pass, unlike the aggregation
    # side's pre-pass, because here a leaf's moment is written during its own iteration above. Safe
    # to defer: nothing later in this chunk writes it (O2O reads the PARENT, a level up and already
    # finished), so reading it after the loop gives the same values as reading it inside.
    #
    # `hasinitializedmoment` re-derives the `initializedmoment` flag rather than the loop storing
    # it -- see its docstring. The flag is a pure function of tree and plan, so this is exact, and it
    # avoids a per-chunk per-matvec flag buffer.
    disaggregateleafchunk!(
        A,
        tree,
        chunknodes,
        y,
        momentcollection,
        disaggregationmoment,
        leafmul!,
        level,
        leafbuffer,
        node -> hasinitializedmoment(disaggregationplan, tree, rootnode, node),
    )
    return nothing
end

"""
    disaggregatetree!(A, plan, schedule, tree, moment, momentcollection,
                      o2otranslation!, y, leafmul!, diagnostics)

Run the scheduled downward pass without I2O translations and scatter leaf moments into `y`.
"""
function disaggregatetree!(
    A,
    disaggregationplan::DisaggregatePlan,
    schedule::PlanSchedule,
    tree,
    moment,
    momentcollection,
    o2otranslation!,
    y::AbstractVector,
    leafmul!,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    # `schedule.nodechunks[i]` is the precomputed, reused vector-of-vectors of disaggregation nodes
    # for `schedule.levels[i]`. Per-level `@tasks` extracted into `_disaggregatelevel!` so `level` /
    # `levelchunks` are arguments (no closure boxing on the no-op path).
    for (level, levelchunks) in zip(schedule.levels, schedule.nodechunks)
        _disaggregatelevel!(
            A,
            disaggregationplan,
            tree,
            moment,
            momentcollection,
            o2otranslation!,
            y,
            leafmul!,
            level,
            levelchunks,
            diagnostics,
        )
    end
    return nothing
end

function _disaggregatelevel!(
    A,
    disaggregationplan,
    tree,
    moment,
    momentcollection,
    o2otranslation!,
    y,
    leafmul!,
    level,
    levelchunks,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    with_timing(diagnostics, :disaggregate, :level; level=level, mode=diagnosticmode(A)) do
        @tasks for chunkid in 1:length(levelchunks)
            @set scheduler = outerscheduler(A, DisaggregateOp(), level)
            disaggregatechunk!(
                A,
                disaggregationplan,
                tree,
                levelchunks[chunkid],
                moment,
                momentcollection,
                o2otranslation!,
                y,
                leafmul!,
                level,
                # `level + 1`: this pass loops PARENTS and pushes each node's moment down to its
                # children, so the transfer it applies is keyed one level below. (The fused
                # translate-mode pass above loops the children themselves, hence no offset there.)
                borrowbuffer(o2ochannel(A), level + 1, chunkid, maptrait(A)),
                # NOT offset, even though the O2O scratch above is: the leaf multiply fires on a node
                # of THIS level (`isleaf(tree, node)` below), not on its children, so its scratch is
                # keyed by that leaf's own level. The two keys genuinely differ in this pass.
                borrowbuffer(disaggregationleafchannel(A), level, chunkid, maptrait(A)),
            )
        end
    end
    return nothing
end

# Serial per-chunk work for the store-mode (O2O-only) downward pass. For each node in the chunk:
# scatter a leaf's moment to its DOFs (`leafmul!`), or push an internal node's moment down to its
# stored children (O2O). `o2otranslation!` applies the disaggregation side's I2I family through an
# explicit transpose/adjoint (`O2O == transpose(I2I)`), so no moment-state conjugation is needed --
# the level transfers carry no hidden conjugation.
"""
    disaggregatechunk!(A, plan, tree, nodes, moment, momentcollection,
                       o2otranslation!, y, leafmul!, level,
                       o2obuffer, leafbuffer)

Push moments down and scatter leaves for one serial chunk of nodes.
"""
function disaggregatechunk!(
    A,
    disaggregationplan,
    tree,
    chunknodes,
    moment,
    momentcollection,
    o2otranslation!,
    y,
    leafmul!,
    level,
    o2obuffer,
    leafbuffer,
)
    # Store-mode kernel: every leaf it visits is written unconditionally, so the predicate handed to
    # `disaggregateleafchunk!` below is a constant `true` -- unlike the fused kernel, which has to
    # re-derive `hasinitializedmoment`.
    for node in chunknodes
        if !H2Trees.isleaf(tree, node)
            for child in H2Trees.children(tree, node)
                if H2Trees.storenode(disaggregationplan, child)
                    o2otranslation!(
                        moment(child),
                        moment(node),
                        tree,
                        node,
                        child,
                        Accumulate(),
                        innerscheduler(A, O2OOp(), level),
                        o2obuffer,
                    )

                else
                    o2otranslation!(
                        moment(child),
                        moment(node),
                        tree,
                        node,
                        child,
                        innerscheduler(A, O2OOp(), level),
                        o2obuffer,
                    )
                end
            end
        end
    end

    disaggregateleafchunk!(
        A,
        tree,
        chunknodes,
        y,
        momentcollection,
        moment,
        leafmul!,
        level,
        leafbuffer,
        Returns(true),
    )
    return nothing
end
