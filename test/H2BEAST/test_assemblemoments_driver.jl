module TestAssembleMomentsDriver

using Test
using StaticArrays
using CompScienceMeshes
using BEAST
using H2Factory

# Tier 1: `ext/H2BEAST/moments/assemblemoments.jl` as PLUMBING. The moment physics is iFMM's; here we
# drive the generic driver with a spy callback + spy buffer factory and assert only what the driver
# itself owns -- cluster traversal, per-cluster data routing, function-id filtering, the
# active/global element mapping, quadrature-data passthrough, and the callback argument order iFMM
# relies on. The active/global mapping is the part most prone to a silent off-by-mapping bug (an
# earlier version called `BEAST.assemblydata` directly and skipped the global->active conversion),
# so it gets both a direct unit test with inactive cells and an end-to-end check.

@testset "globaltoactiveelements: inverts the active->global map, zero for inactive cells" begin
    # `BEAST.assemblydata` returns active->global (active index i -> its global cell id). The driver
    # indexes by GLOBAL cell id, so it needs the inverse: global cell id -> active index (0 where the
    # cell carries no basis function). Synthetic maps with gaps exercise exactly the inactive-cell
    # case a full-support basis can never reach.
    #                          active 1->cell2, 2->cell3, 3->cell5 ; cells 1,4 inactive
    g2a = H2Factory.globaltoactiveelements([2, 3, 5], 5)
    @test g2a == [0, 1, 2, 0, 3]
    # round-trip: every active id maps back to its global cell
    for (activeid, globalid) in enumerate([2, 3, 5])
        @test g2a[globalid] == activeid
    end

    # All cells active -> identity, returned as the range itself (the fast path).
    @test H2Factory.globaltoactiveelements(1:4, 4) === 1:4
    @test H2Factory.globaltoactiveelements([1, 2, 3, 4], 4) == 1:4
end

@testset "assemblydata returns the global->active map (not BEAST's active->global)" begin
    # The whole point of `H2Factory.assemblydata` over a raw `BEAST.assemblydata` call: its third
    # return is the global->active map the driver indexes by. Verify it equals the conversion of
    # BEAST's active->global, so the driver entry can rely on it.
    m = meshrectangle(1.0, 1.0, 0.5)
    X = lagrangecxd0(m)                      # one DOF per cell -> every cell active here
    elements, ad, g2a = H2Factory.assemblydata(X; onlyactives=true)
    _, _, activetoglobal = BEAST.assemblydata(X; onlyactives=true)
    @test g2a == H2Factory.globaltoactiveelements(activetoglobal, BEAST.numcells(m))
    @test length(elements) == length(activetoglobal)
end

@testset "driver plumbing: visit-once, per-cluster routing, function-id filter, qdata, arg order" begin
    # Drive the real entry with a spy callback + spy buffer factory over an all-active basis. Partition
    # the basis functions into fake "leaf clusters" and assert the driver visits each exactly once
    # (multithreaded, no skip/dup), routes `clusterdata[cluster]` to both the buffer factory and the
    # callback, filters function ids per cluster, builds+passes `qdata`, and hands the callback its
    # arguments in the documented order (checked by asserting known values land in known positions).
    m = meshrectangle(1.0, 1.0, 0.25)
    X = lagrangecxd0(m)                       # one DOF per cell -> all cells active
    nf = numfunctions(X)
    clusters = [collect(g) for g in Iterators.partition(1:nf, 3)]
    clusterdata = ["cluster-$i" for i in eachindex(clusters)]
    op = Helmholtz3D.singlelayer(; wavenumber=1.0)
    _, _, g2a = H2Factory.assemblydata(X)

    lk = ReentrantLock()
    records = Dict{Int,NamedTuple}()

    spybuffer(operators, basis, refspace, clusterdatum, type) = (:buffer, clusterdatum)
    function spycallback(
        store,
        operators,
        basis,
        fids,
        refspace,
        cbuf,
        elements,
        active,
        ad,
        qdata,
        qs,
        cdatum,
        cluster,
    )
        lock(lk) do
            records[cluster] = (;
                store,
                operators,
                basis,
                fids,
                cbuf,
                active=copy(active),
                qdatalen=length(qdata),
                cdatum,
            )
        end
        return nothing
    end

    H2Factory.assemblemoments!(
        :store,
        op,
        X,
        clusters,
        clusterdata,
        Val(ComplexF64),
        BEAST.SingleNumQStrat(2),
        spycallback,
        spybuffer,
    )

    @test sort(collect(keys(records))) == collect(1:length(clusters))   # each cluster once
    for c in 1:length(clusters)
        r = records[c]
        @test r.cdatum == clusterdata[c]                 # per-cluster data routed to callback
        @test r.cbuf == (:buffer, clusterdata[c])        # ...and to the buffer factory
        @test r.fids == Set(clusters[c])                 # function ids filtered per cluster
        expected = sort!(
            unique!([g2a[el.cellid] for fid in clusters[c] for el in X.fns[fid]])
        )
        @test r.active == expected                       # active elements correctly gathered
        @test r.qdatalen > 0                             # qdata built and passed
        # arg order: known objects in their documented positions
        @test r.store === :store
        @test r.operators === op
        @test r.basis === X
    end
end

@testset "inactive cells: active-element mapping is correct (discriminates the assemblydata fix)" begin
    # The discriminating fixture. A subset space over SCATTERED cells (not a prefix) makes the
    # active->global and global->active maps genuinely differ on the active cells themselves, so the
    # driver's `globaltoactiveelements[element.cellid]` is only correct if the entry used
    # `H2Factory.assemblydata` (the converted map). With the pre-fix `BEAST.assemblydata` call this
    # errors (index past the active-length vector) or mismaps -- i.e. this test is RED on the old code.
    m = meshrectangle(1.0, 1.0, 0.25)
    @test numcells(m) >= 6
    scattered = [2, 4, 5]
    X = lagrangecxd0(m)
    Xsub = BEAST.subset(X, scattered)         # functions on cells 2,4,5; the rest inactive
    _, _, g2a = H2Factory.assemblydata(Xsub)
    @test any(iszero, g2a)                     # there really ARE inactive cells (else vacuous)

    nf = numfunctions(Xsub)
    clusters = [[i] for i in 1:nf]
    clusterdata = collect(1:nf)
    op = Helmholtz3D.singlelayer(; wavenumber=1.0)

    lk = ReentrantLock()
    activebycluster = Dict{Int,Vector{Int}}()
    spybuffer(args...) = :buf
    function spycallback(
        store,
        operators,
        basis,
        fids,
        refspace,
        cbuf,
        elements,
        active,
        ad,
        qdata,
        qs,
        cdatum,
        cluster,
    )
        lock(lk) do
            activebycluster[cluster] = copy(active)
        end
        return nothing
    end

    H2Factory.assemblemoments!(
        :store,
        op,
        Xsub,
        clusters,
        clusterdata,
        Val(ComplexF64),
        BEAST.SingleNumQStrat(2),
        spycallback,
        spybuffer,
    )

    for i in 1:nf
        expected = sort!(unique!([g2a[el.cellid] for el in Xsub.fns[i]]))
        @test activebycluster[i] == expected
        @test all(>(0), activebycluster[i])    # valid active indices, never the inactive sentinel 0
    end
end

end # module TestAssembleMomentsDriver
