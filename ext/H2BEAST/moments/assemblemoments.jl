# Convert global cell IDs to active-element indices, then remove duplicates and sort.
function activeelementids(functionids, basis, globaltoactiveelements)
    ids = Int[]
    for m in functionids
        for element in basis.fns[m]
            activeid = globaltoactiveelements[element.cellid]
            activeid == 0 &&
                error("Basis function $m references inactive cell $(element.cellid).")
            push!(ids, activeid)
        end
    end
    sort!(ids)
    return unique!(ids)
end

# Build one cluster's buffer and quadrature data, then call `assemblelocal!`.
function assemblecluster!(
    store,
    operators,
    basis,
    functionids,
    refspace,
    type,
    buffer,
    elements,
    assemblydata,
    globaltoactiveelements,
    clusterdatum,
    cluster,
    quadstrat,
    assemblelocal!,
)
    computationbuffer = buffer(operators, basis, refspace, clusterdatum, type)
    activeelements = activeelementids(functionids, basis, globaltoactiveelements)

    qdata = H2Factory.quaddata(
        operators, refspace, basis, elements[activeelements], quadstrat
    )

    assemblelocal!(
        store,
        operators,
        basis,
        Set(functionids),
        refspace,
        computationbuffer,
        elements,
        activeelements,
        assemblydata,
        qdata,
        quadstrat,
        clusterdatum,
        cluster,
    )

    return nothing
end

"""
    assemblemoments!(store, operators, basis, clusters, clusterdata, type,
                    quadstrat, assemblelocal!, buffer; verbose=false)

Assemble moments for clusters of basis-function IDs. Each cluster gets a computation buffer from
`buffer(operators, basis, refspace, clusterdata[cluster], type)` and quadrature data for its active
elements. The callback `assemblelocal!` writes the result to `store`.

Clusters are processed in parallel, so callbacks must support concurrent calls. Returns `nothing`.
Requires BEAST.
"""
function H2Factory.assemblemoments!(
    store,
    operators,
    basis,
    clusters::AbstractVector,
    clusterdata::AbstractArray,
    type,
    quadstrat,
    assemblelocal!,
    buffer;
    verbose=false,
)
    # Use the global-to-active map; BEAST returns the opposite mapping.
    elements, assemblydata, globaltoactiveelements = H2Factory.assemblydata(
        basis; onlyactives=true
    )
    refspace = BEAST.refspace(basis)

    pbar = progressbar(Threads.nthreads(), verbose)

    @tasks for taskclusters in chunks(1:length(clusters); n=Threads.nthreads())
        H2Factory.assemblemoments!(
            store,
            operators,
            basis,
            clusters,
            refspace,
            type,
            buffer,
            elements,
            assemblydata,
            globaltoactiveelements,
            clusterdata,
            taskclusters,
            quadstrat,
            assemblelocal!,
        )
        next!(pbar)
    end
    finish!(pbar)

    return nothing
end

# Process one task's clusters. Each cluster gets its own buffer and quadrature data.
function H2Factory.assemblemoments!(
    store,
    operators,
    basis,
    functionidsclusters::AbstractVector,
    refspace,
    type,
    buffer,
    elements,
    assemblydata,
    globaltoactiveelements,
    clusterdata::AbstractArray,
    clusters::AbstractVector,
    quadstrat,
    assemblelocal!,
)
    for cluster in clusters
        assemblecluster!(
            store,
            operators,
            basis,
            functionidsclusters[cluster],
            refspace,
            type,
            buffer,
            elements,
            assemblydata,
            globaltoactiveelements,
            clusterdata[cluster],
            cluster,
            quadstrat,
            assemblelocal!,
        )
    end

    return nothing
end

function H2Factory.quaddata(
    operator, refspace, basis, elements, quadstrat::BEAST.SingleNumQStrat
)
    return BEAST.quadpoints(refspace, elements, (quadstrat.quad_rule,))
end

"""
    assemblydata(basis; onlyactives=true)

Return `(elements, assemblydata, globaltoactiveelements)` for a BEAST basis.
The third value maps global cell IDs to indices in `elements`, with zero for inactive cells.
This reverses the active-to-global map returned by `BEAST.assemblydata`.
"""
function H2Factory.assemblydata(basis; onlyactives=true)
    elements, assemblydata, activetoglobalelements = BEAST.assemblydata(
        basis; onlyactives=onlyactives
    )

    globaltoactiveelements = H2Factory.globaltoactiveelements(
        activetoglobalelements, BEAST.numcells(geometry(basis))
    )

    return elements, assemblydata, globaltoactiveelements
end

"""
    globaltoactiveelements(activetoglobalelements, numcells)

Invert an active-to-global cell map. The result has `numcells` entries, with zero for inactive
cells. When the map is the identity, return `1:numcells`.
"""
function H2Factory.globaltoactiveelements(activetoglobalelements, numcells::Int)
    activetoglobalelements == 1:numcells && return 1:numcells

    globaltoactiveelements = zeros(Int, numcells)
    for (activeid, globalid) in enumerate(activetoglobalelements)
        globaltoactiveelements[globalid] = activeid
    end
    return globaltoactiveelements
end
