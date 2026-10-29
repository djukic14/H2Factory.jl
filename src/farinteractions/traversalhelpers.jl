# Shared per-node work for the aggregation and disaggregation traversals.

"""
    upwardpass!(A, tree, node, moment, momentcollection, i2itranslation!, leafmul!, x, level,
                i2ibuffer, leafbuffer)

Process one node in the upward pass. Leaves apply their leaf matrix; internal nodes aggregate their
children with I2I translations.
"""
function upwardpass!(
    A,
    tree,
    node,
    moment,
    momentcollection,
    i2itranslation!,
    leafmul!,
    x,
    level,
    i2ibuffer,
    leafbuffer,
)
    if H2Trees.isleaf(tree, node)
        aggregateleaf!(A, tree, node, moment, momentcollection, leafmul!, x, leafbuffer)
    else
        i2ipass!(A, tree, node, moment, i2itranslation!, level, i2ibuffer)
    end
    return nothing
end

"""
    aggregateleaf!(A, tree, node, moment, momentcollection, leafmul!, x, leafbuffer)

Gather a leaf's coefficients from `x` and multiply its leaf collection into its moment.
"""
function aggregateleaf!(A, tree, node, moment, momentcollection, leafmul!, x, leafbuffer)
    aggregationvectorbuffer(A, node) .= view(x, H2Trees.values(tree, node))
    leafmul!(
        leafbuffer, moment(node), momentcollection(node), aggregationvectorbuffer(A, node)
    )
    return nothing
end

"""
    i2ipass!(A, tree, node, moment, i2itranslation!, level, i2ibuffer)

Combine the moments of all children into the moment of `node`.
"""
function i2ipass!(A, tree, node, moment, i2itranslation!, level, i2ibuffer)
    for (i, child) in enumerate(H2Trees.children(tree, node))
        if i == 1
            i2itranslation!(
                moment(node),
                moment(child),
                tree,
                node,
                child,
                innerscheduler(A, I2IOp(), level),
                i2ibuffer,
            )
        else
            i2itranslation!(
                moment(node),
                moment(child),
                tree,
                node,
                child,
                Accumulate(),
                innerscheduler(A, I2IOp(), level),
                i2ibuffer,
            )
        end
    end
    return nothing
end

"""
    disaggregateleaf!(A, tree, node, y, momentcollection, moment, leafmul!, leafbuffer)

Scatter a leaf's incoming moment onto its degrees of freedom in `y`.

The map's scale `α(A)` is applied and the destination slice is overwritten.
"""
function disaggregateleaf!(A, tree, node, y, momentcollection, moment, leafmul!, leafbuffer)
    disaggregationvectorbuffer(A, node) .= view(y, H2Trees.values(tree, node))
    leafmul!(
        leafbuffer,
        disaggregationvectorbuffer(A, node),
        momentcollection(node),
        moment(node),
        α(A),
        false,
    )
    view(y, H2Trees.values(tree, node)) .= disaggregationvectorbuffer(A, node)
    return nothing
end

"""
    hasinitializedmoment(disaggregationplan, tree, rootnode, node) -> Bool

Whether the fused downward pass will have written `node`'s disaggregation moment.
"""
function hasinitializedmoment(disaggregationplan, tree, rootnode, node)
    if node != rootnode &&
        H2Trees.isdisaggregationnode(disaggregationplan, H2Trees.parent(tree, node))
        return true
    end
    for _ in H2Trees.translatingnodes(disaggregationplan, node)
        return true
    end
    return false
end

# --- chunk-level leaf hooks -----------------------------------------------------------------------
#
# Chunk hooks dispatch on the leaf-buffer type so implementations may batch leaf work. The default
# methods process one leaf at a time.

# `aggregateleafchunk!` is DECLARED in `src/Interfaces.jl`, with its contract. This is H2Factory's
# fallback: one `aggregateleaf!` per node.
function aggregateleafchunk!(
    A, tree, chunknodes, moment, momentcollection, leafmul!, x, level, leafbuffer
)
    for node in chunknodes
        H2Trees.isleaf(tree, node) || continue
        aggregateleaf!(A, tree, node, moment, momentcollection, leafmul!, x, leafbuffer)
    end
    return nothing
end

"""
    disaggregateleafchunk!(A, tree, chunknodes, y, momentcollection, moment, leafmul!, level,
                           leafbuffer, initialized)

Scatter every written leaf in `chunknodes` onto its degrees of freedom.

`initialized(node)` selects the leaves whose moments are ready. The default processes each selected
leaf with [`disaggregateleaf!`](@ref).
"""
function disaggregateleafchunk!(
    A,
    tree,
    chunknodes,
    y,
    momentcollection,
    moment,
    leafmul!,
    level,
    leafbuffer,
    initialized,
)
    for node in chunknodes
        H2Trees.isleaf(tree, node) || continue
        initialized(node) || continue
        disaggregateleaf!(A, tree, node, y, momentcollection, moment, leafmul!, leafbuffer)
    end
    return nothing
end

"""
    translateintonode!(A, tree, receivingnode, translatingnodes, targetmoment,
                       sourcemoment, translatei2o!, level, scheduler, i2obuffer) -> Bool

Translate all `translatingnodes` into `receivingnode` and reverse its I2O modulation. Return `true`
when at least one translation was applied.
"""
function translateintonode!(
    A,
    tree,
    receivingnode,
    translatingnodes,
    targetmoment,
    sourcemoment,
    translatei2o!,
    level,
    threading,
    i2obuffer,
)
    initialized = false
    for translatingnode in translatingnodes
        if initialized
            translatei2o!(
                targetmoment(receivingnode),
                sourcemoment(translatingnode),
                tree,
                receivingnode,
                translatingnode,
                Accumulate(),
                threading,
                i2obuffer,
            )
        else
            translatei2o!(
                targetmoment(receivingnode),
                sourcemoment(translatingnode),
                tree,
                receivingnode,
                translatingnode,
                threading,
                i2obuffer,
            )
            initialized = true
        end
    end
    if initialized
        # Reads its own `ReverseModulationOp` scheduler rather than reusing `threading` (the I2O
        # one): it is a distinct kernel -- the inverse-FFT/demodulation step undoing the
        # `modulate_in!` applied on the aggregation side -- so its cost profile is its own, and
        # `InnerScheduling` gives it its own field.
        modulate_out!(
            targetmoment(receivingnode),
            level,
            A,
            innerscheduler(A, ReverseModulationOp(), level),
            i2obuffer,
        )
    end
    return initialized
end
