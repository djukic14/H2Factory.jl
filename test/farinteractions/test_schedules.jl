module TestSchedules

using Test
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils

# Unit tests for the precomputed chunk schedules (`PlanSchedule`/`Schedules`, `planschedule`, the
# `aggregationschedule`/`disaggregationschedule` accessors) that the matvec traversal now consumes in
# place of the per-MVP `collect` + `index_chunks`. The load-bearing invariant: the stored,
# vector-of-vectors chunk schedule a map serves must, when its chunks are concatenated in order,
# reproduce *exactly* the live node order the traversal used to compute fresh, for the plan each
# orientation (forward / transpose / adjoint) actually resolves to. If that holds, swapping the live
# computation for the cache is behavior-preserving (the far-interactions harness confirms the numeric
# result separately).

# Flatten a PlanSchedule's per-level chunks back into one node vector per level (chunk order preserved).
_flat(sched) =
    [reduce(vcat, sched.nodechunks[i]; init=Int[]) for i in eachindex(sched.levels)]
_flatrecv(sched) =
    [reduce(vcat, sched.receivingchunks[i]; init=Int[]) for i in eachindex(sched.levels)]

# Assert a map's aggregation/disaggregation schedules match the live node order of the plans that same
# map (in whatever orientation) resolves to. `M` may be `A`, `transpose(A)`, or `adjoint(A)`.
function check_schedule_matches_plans(M)
    aggplan = H2Factory.aggregationplan(M)
    aggsched = H2Factory.aggregationschedule(M)
    @test aggsched.levels == collect(H2Trees.aggregationlevels(aggplan))
    for (i, level) in enumerate(aggsched.levels)
        @test _flat(aggsched)[i] == collect(H2Trees.aggregationnodes(aggplan, level))
    end
    # receiving chunks only exist for the fused AggregateTranslatePlan
    if aggplan isa H2Trees.AggregateTranslatePlan
        for (i, level) in enumerate(aggsched.levels)
            @test _flatrecv(aggsched)[i] == collect(H2Trees.receivingnodes(aggplan, level))
        end
    else
        @test aggsched.receivingchunks === nothing
    end

    disaggplan = H2Factory.disaggregationplan(M)
    disaggsched = H2Factory.disaggregationschedule(M)
    @test disaggsched.levels == collect(H2Trees.disaggregationlevels(disaggplan))
    for (i, level) in enumerate(disaggsched.levels)
        @test _flat(disaggsched)[i] ==
            collect(H2Trees.disaggregationnodes(disaggplan, level))
    end
    return nothing
end

# Uniform points + small halfsize -> a deep tree with genuine far pairs (a large halfsize gives an
# empty AggregatePlan, which H2Trees rejects). Matches the harness's tree recipe.
uniformpts(N) = [(i + 0.5) / N for i in 0:(N - 1)]
mkpts(xs) = [(x, 0.0, 0.0) for x in xs]

# A map that stores an `ExecutionScheduling` and NO `scheduler` field, the shape iFMM's
# `GalerkiniFMM`/`PetrovGalerkiniFMM` actually have. This is what makes the regression below real: the
# stock fakes carry a `scheduler` field, so for them the uniform fallback happens to succeed and the
# bug stays invisible. Here the fallback is a `FieldError`, so a skipped override cannot hide.
# Stores ONE bare scheduler and nothing else, the subject for H2Factory's UNIFORM fallback
# (`outerscheduler(A, ::AbstractMapTrait, ::AbstractH2Operation, level) = scheduler(A)`), which is
# what a map that has not opted into per-plan scheduling gets. The test fakes used to serve this
# role, but they are `H2Workspace`s now and carry a real `ExecutionScheduling`, so the fallback needs
# its own minimal map or it goes untested.
struct SchedulerOnlyMap{T,S} <: H2Factory.GalerkinH2Map{T}
    nearinteractions::Matrix{T}   # only `size(A)` reads it
    scheduler::S
end

struct SchedulingOnlyMap{T,S} <: H2Factory.GalerkinH2Map{T}
    nearinteractions::Matrix{T}   # only `size(A)` reads it
    scheduling::S
end

# Deliberately written at the trait level, exactly as a real map must (see the testset below), and
# forwarding the trait into the plan-keyed `OuterScheduling` rather than dropping it, mirrors iFMM's
# override exactly.
H2Factory.outerscheduler(
    A::SchedulingOnlyMap,
    trait::H2Factory.AbstractMapTrait,
    op::H2Factory.AbstractH2Operation,
    level::Int,
) = H2Factory.outerscheduler(A.scheduling.outer, op, trait, level)
H2Factory.innerscheduler(
    A::SchedulingOnlyMap,
    ::H2Factory.AbstractMapTrait,
    op::H2Factory.AbstractH2Operation,
    level::Int,
) = H2Factory.innerscheduler(A.scheduling.inner, op, level)

@testset "Galerkin schedule reproduces live node order (fwd/transpose/adjoint x modes)" begin
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        A = fakegalerkinmap(
            tree; T=ComplexF64, i2o=scalartranslator(1.0 + 0.0im), aggregationmode=mode
        )
        check_schedule_matches_plans(A)
        check_schedule_matches_plans(transpose(A))
        check_schedule_matches_plans(adjoint(A))
    end
end

@testset "Petrov schedule reproduces live node order (fwd/transpose/adjoint x modes)" begin
    # overlapping test/trial regions so the four Petrov plans genuinely differ between sides
    u = uniformpts(128)
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        A = fakepetrovmap(
            mkpts(u),
            mkpts(0.5 .+ u);
            T=ComplexF64,
            i2o=scalartranslator(1.0 + 0.0im),
            minhalfsize=0.02,
            aggregationmode=mode,
        )
        check_schedule_matches_plans(A)
        check_schedule_matches_plans(transpose(A))
        check_schedule_matches_plans(adjoint(A))
    end
end

@testset "stored schedule is served (not the on-the-fly fallback)" begin
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    A = fakegalerkinmap(tree; T=ComplexF64, i2o=scalartranslator(1.0 + 0.0im))
    # The override serves the *identical stored object* each call (=== ), whereas the generic fallback
    # would build a fresh PlanSchedule every time. `===` pins that the cache is actually reused.
    @test H2Factory.aggregationschedule(A) === H2Factory.aggregationschedule(A)
    @test H2Factory.disaggregationschedule(A) === H2Factory.disaggregationschedule(A)
end

@testset "Petrov serves stored schedules (not the on-the-fly fallback)" begin
    # Mirrors the Galerkin cache test. A map that stores `schedules` but forgets the accessor
    # overrides silently falls back to rebuilding from the plan on every matvec, and the rebuild
    # uses the *current* `nchunks()`, which for a map built with `nchunks < nthreads` yields more
    # chunks than there are buffers (an out-of-bounds `borrowbuffer(..., chunkid)`). `===` pins that
    # the stored object is actually served, in every orientation.
    u = uniformpts(128)
    A = fakepetrovmap(
        mkpts(u),
        mkpts(0.5 .+ u);
        T=ComplexF64,
        i2o=scalartranslator(1.0 + 0.0im),
        minhalfsize=0.02,
    )
    for M in (A, transpose(A), adjoint(A))
        @test H2Factory.aggregationschedule(M) === H2Factory.aggregationschedule(M)
        @test H2Factory.disaggregationschedule(M) === H2Factory.disaggregationschedule(M)
    end
    # transpose flips to the other side's stored schedule, not a rebuild of the same one
    @test H2Factory.aggregationschedule(transpose(A)) !== H2Factory.aggregationschedule(A)
end

@testset "checkexecutiondata: records build nchunks, no warn when consistent" begin
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    A = fakegalerkinmap(tree; T=ComplexF64, i2o=scalartranslator(1.0 + 0.0im))
    # the schedule records the thread count it was materialized for (== nchunks() at construction)
    @test H2Factory.aggregationschedule(A).nchunks == Threads.nthreads()
    # built for the current thread count -> guard is a no-op, no warning
    @test H2Factory.checkexecutiondata(A) === nothing
    @test_nowarn H2Factory.checkexecutiondata(A)
    # a fresh schedule built for a deliberately different thread count is what would trip the warn
    @test H2Factory.planschedule(H2Factory.aggregationplan(A), Threads.nthreads() + 3).nchunks ==
        Threads.nthreads() + 3
end

@testset "ExecutionScheduling: outer per plan/level, inner per kernel/level" begin
    lv = 1:5
    s = OhMyThreads.SerialScheduler()
    # production default: threaded outer on every plan/level, serial inner on every kernel/level
    e = defaultexecutionscheduling(lv)
    for op in (AggregateOp(), I2OOp(), DisaggregateOp()), l in lv
        @test outerscheduler(e.outer, op, l) isa OhMyThreads.DynamicScheduler
    end
    for op in
        (I2IOp(), I2OOp(), O2OOp(), ModulationOp(), ReverseModulationOp(), LeafMulOp()),
        l in lv

        @test innerscheduler(e.inner, op, l) isa OhMyThreads.SerialScheduler
    end
    # `leafmul` is the ONE inner kernel a backend can pre-set parallel, and it must reach the
    # container without disturbing any other kernel, a BATCHED leaf multiply (MLFMA's compressed
    # leaf storages) has a per-group projection loop whose serial default would be a 4x regression,
    # while every other inner kernel stays serial by default. So assert both halves: the override
    # lands, and nothing else moved.
    leaf = OhMyThreads.DynamicScheduler(; ntasks=3)
    eleaf = defaultexecutionscheduling(lv; leafmul=leaf)
    for l in lv
        @test innerscheduler(eleaf.inner, LeafMulOp(), l) === leaf
        for op in (I2IOp(), I2OOp(), O2OOp(), ModulationOp(), ReverseModulationOp())
            @test innerscheduler(eleaf.inner, op, l) isa OhMyThreads.SerialScheduler
        end
    end
    # ...and it reaches through `executionscheduling`, the entry point a map constructor actually
    # calls (MLFMA's `executionschedulingfor` goes through that, not through the default directly).
    ethrough = executionscheduling(OhMyThreads.DynamicScheduler(), lv; leafmul=leaf)
    @test innerscheduler(ethrough.inner, LeafMulOp(), 3) === leaf
    # Unset, it stays serial like every other inner kernel: the plumbing adds an OPTION, not a new
    # default. `serialexecutionscheduling` is fully serial by construction and must stay so.
    @test innerscheduler(defaultexecutionscheduling(lv).inner, LeafMulOp(), 3) isa
        OhMyThreads.SerialScheduler
    @test innerscheduler(serialinnerscheduling(lv), LeafMulOp(), 3) isa
        OhMyThreads.SerialScheduler
    @test innerscheduler(serialexecutionscheduling(lv).inner, LeafMulOp(), 3) isa
        OhMyThreads.SerialScheduler

    # capping the traversal at n workers is an outer-scheduler choice, independent of nchunks: the
    # user's scheduler object is what every plan/level serves
    cap = OhMyThreads.DynamicScheduler(; ntasks=2)
    ecap = defaultexecutionscheduling(lv; outer=cap)
    for op in (AggregateOp(), I2OOp(), DisaggregateOp())
        @test outerscheduler(ecap.outer, op, 1) === cap
    end

    # fully serial (debugging / determinism A-B)
    ser = serialexecutionscheduling(lv)
    for op in (AggregateOp(), I2OOp(), DisaggregateOp())
        @test outerscheduler(ser.outer, op, 2) isa OhMyThreads.SerialScheduler
    end

    # sparse per-(plan, level) overrides are baked into the containers at build time, so the
    # traversal never pays a fallback lookup. Overrides name a *plan*, since that is what a container
    # holds, the four (phase, orientation) combinations are four distinct node sets.
    o = outerscheduling(
        OhMyThreads.DynamicScheduler(),
        lv,
        (I2OOp(), 3) => s,
        (TestDisaggregationOp(), 5) => s,
    )
    @test outerscheduler(o, I2OOp(), 3) === s                    # exact cell hit
    @test outerscheduler(o, TestDisaggregationOp(), 5) === s
    # ...and the phase tag reaches the same cell: forward DisaggregateOp walks the test disaggregation
    # nodes, so it must resolve to the container that override landed in
    @test outerscheduler(o, DisaggregateOp(), H2Factory.isForward(), 5) === s
    # ...while transposed it walks the *trial* disaggregation nodes, a different node set, untouched
    @test outerscheduler(o, DisaggregateOp(), H2Factory.isTranspose(), 5) isa
        OhMyThreads.DynamicScheduler
    @test outerscheduler(o, I2OOp(), 4) isa OhMyThreads.DynamicScheduler   # other level
    @test outerscheduler(o, AggregateOp(), 3) isa OhMyThreads.DynamicScheduler # other plan

    # a bare scheduler is accepted where an ExecutionScheduling is expected; a built one passes through
    @test executionscheduling(OhMyThreads.DynamicScheduler(), lv) isa ExecutionScheduling
    @test executionscheduling(e, lv) === e
end

@testset "outer schedulers are keyed by plan: same node set -> same scheduler" begin
    # Outer scheduling follows the stored traversal *plans*, not the phase: the four
    # (phase, orientation) combinations walk four distinct node sets, so each gets its own container.
    # Distinct scheduler objects per field make the resolution observable.
    lv = 1:5
    trialagg = OhMyThreads.DynamicScheduler(; ntasks=2)
    testagg = OhMyThreads.DynamicScheduler(; ntasks=3)
    i2o = OhMyThreads.DynamicScheduler(; ntasks=4)
    testdisagg = OhMyThreads.DynamicScheduler(; ntasks=5)
    trialdisagg = OhMyThreads.DynamicScheduler(; ntasks=6)
    outer = OuterScheduling(
        perlevel(trialagg, lv),
        perlevel(testagg, lv),
        perlevel(i2o, lv),
        perlevel(testdisagg, lv),
        perlevel(trialdisagg, lv),
    )
    A = SchedulingOnlyMap(
        zeros(ComplexF64, 4, 4),
        ExecutionScheduling(outer, serialinnerscheduling(lv), Threads.nthreads()),
    )

    # The four traversals, each resolving to its own plan's scheduler.
    @test H2Factory.outerscheduler(A, AggregateOp(), 3) === trialagg
    @test H2Factory.outerscheduler(A, DisaggregateOp(), 3) === testdisagg
    for M in (transpose(A), adjoint(A))
        @test H2Factory.outerscheduler(M, AggregateOp(), 3) === testagg
        @test H2Factory.outerscheduler(M, DisaggregateOp(), 3) === trialdisagg
    end

    # The invariant, stated as the pairing it guarantees: each of the four node sets has exactly one
    # scheduler, and no two traversals share one.
    resolved = [
        H2Factory.outerscheduler(A, AggregateOp(), 3),
        H2Factory.outerscheduler(A, DisaggregateOp(), 3),
        H2Factory.outerscheduler(transpose(A), AggregateOp(), 3),
        H2Factory.outerscheduler(transpose(A), DisaggregateOp(), 3),
    ]
    @test length(unique(objectid.(resolved))) == 4

    # i2o has no orientation partner: the fused receiving-node loop serves both directions
    for M in (A, transpose(A), adjoint(A))
        @test H2Factory.outerscheduler(M, I2OOp(), 3) === i2o
    end

    # plan tags address a field directly, orientation-free
    @test outerscheduler(outer, TrialAggregationOp(), 3) === trialagg
    @test outerscheduler(outer, TestAggregationOp(), 3) === testagg
    @test outerscheduler(outer, TestDisaggregationOp(), 3) === testdisagg
    @test outerscheduler(outer, TrialDisaggregationOp(), 3) === trialdisagg
    # the phase-tag compatibility form means forward
    @test outerscheduler(outer, AggregateOp(), 3) === trialagg
    @test outerscheduler(outer, DisaggregateOp(), 3) === testdisagg
end

@testset "inner transfer schedulers follow the tree, flipping under transpose/adjoint" begin
    # The transfer kernels are tied to tree sides, and the sides swap with orientation: the upward pass
    # walks the trial tree forward but the *test* tree under transpose. So the I2I/O2O inner-scheduler
    # *lookup* flips with the map, keeping a scheduler attached to the same tree in every orientation.
    # The executed kernel does not flip (the upward pass always calls the I2I translator; `O2O` is built
    # as `permutedims(I2I)`, so the upward pass's mathematical `O2Oᵀ` *is* `I2I`), only which field is
    # read. Distinct scheduler objects per field make the flip observable; the all-serial production
    # default would hide it.
    lv = 1:5
    i2isched = OhMyThreads.DynamicScheduler(; ntasks=2)
    o2osched = OhMyThreads.DynamicScheduler(; ntasks=3)
    i2osched = OhMyThreads.SerialScheduler()
    modsched = OhMyThreads.DynamicScheduler(; ntasks=4)
    revsched = OhMyThreads.DynamicScheduler(; ntasks=5)
    leafsched = OhMyThreads.DynamicScheduler(; ntasks=6)
    inner = InnerScheduling(
        perlevel(i2isched, lv),
        perlevel(i2osched, lv),
        perlevel(o2osched, lv),
        perlevel(modsched, lv),
        perlevel(revsched, lv),
        perlevel(leafsched, lv),
    )
    outer = OuterScheduling(
        perlevel(OhMyThreads.DynamicScheduler(), lv),
        perlevel(OhMyThreads.DynamicScheduler(), lv),
        perlevel(OhMyThreads.DynamicScheduler(), lv),
        perlevel(OhMyThreads.DynamicScheduler(), lv),
        perlevel(OhMyThreads.DynamicScheduler(), lv),
    )
    A = SchedulingOnlyMap(
        zeros(ComplexF64, 4, 4), ExecutionScheduling(outer, inner, Threads.nthreads())
    )

    # forward: each tag reads its own field
    @test H2Factory.innerscheduler(A, I2IOp(), 3) === i2isched
    @test H2Factory.innerscheduler(A, O2OOp(), 3) === o2osched

    # the invariant: same tree -> same scheduler, in either direction
    for M in (transpose(A), adjoint(A))
        @test H2Factory.innerscheduler(M, I2IOp(), 3) === o2osched
        @test H2Factory.innerscheduler(M, O2OOp(), 3) === i2isched
        # stated as the round-trip identity the flip exists to guarantee
        @test H2Factory.innerscheduler(A, O2OOp(), 3) ===
            H2Factory.innerscheduler(M, I2IOp(), 3)
        @test H2Factory.innerscheduler(A, I2IOp(), 3) ===
            H2Factory.innerscheduler(M, O2OOp(), 3)
    end

    # single-sided kernels do NOT flip: the translation and its modulation pair are the same kernel in
    # either direction (the I2O translation handles orientation itself, via Forward/TransposeTranslation)
    for M in (A, transpose(A), adjoint(A))
        @test H2Factory.innerscheduler(M, I2OOp(), 3) === i2osched
        @test H2Factory.innerscheduler(M, ModulationOp(), 3) === modsched
        @test H2Factory.innerscheduler(M, ReverseModulationOp(), 3) === revsched
        # `LeafMulOp` does not flip either, but NOT because the kernel is single-sided, the leaf
        # multiply really does change side with orientation. It cannot flip because there is only one
        # `leafmul` field, so it has no partner to flip to. Same observable, different reason.
        @test H2Factory.innerscheduler(M, LeafMulOp(), 3) === leafsched
    end

    # the resolver itself, directly
    @test H2Factory.scheduledop(I2IOp(), H2Factory.isForward()) isa I2IOp
    @test H2Factory.scheduledop(I2IOp(), H2Factory.isTranspose()) isa O2OOp
    @test H2Factory.scheduledop(O2OOp(), H2Factory.isAdjoint()) isa I2IOp
    @test H2Factory.scheduledop(I2OOp(), H2Factory.isTranspose()) isa I2OOp
end

@testset "ExecutionScheduling owns nchunks" begin
    # `nchunks` (node-partition granularity = buffer-pool size) has a single owner: the scheduling. A
    # constructor reads it back with `nchunks(scheduling)` and sizes BOTH the chunk schedule and every
    # buffer pool from it, so the two cannot drift apart (a schedule with more chunks than the pool has
    # buffers is an out-of-bounds `buffer[chunkid]`).
    lv = 1:5
    @test H2Factory.nchunks(defaultexecutionscheduling(lv)) == Threads.nthreads()
    @test H2Factory.nchunks(defaultexecutionscheduling(lv; nchunks=7)) == 7
    # a serial traversal has no reason to partition, so it defaults to a single chunk
    @test H2Factory.nchunks(serialexecutionscheduling(lv)) == 1
    @test H2Factory.nchunks(serialexecutionscheduling(lv; nchunks=4)) == 4
    # nchunks is independent of the worker cap: capping workers is an outer-scheduler choice
    capped = defaultexecutionscheduling(
        lv; outer=OhMyThreads.DynamicScheduler(; ntasks=2), nchunks=16
    )
    @test H2Factory.nchunks(capped) == 16
    # a prebuilt scheduling carries its own nchunks through `executionscheduling` untouched
    @test H2Factory.nchunks(executionscheduling(capped, lv)) == 16
    # the bare-scheduler path takes nchunks as a keyword, defaulting to one chunk per thread
    @test H2Factory.nchunks(executionscheduling(OhMyThreads.DynamicScheduler(), lv)) ==
        Threads.nthreads()
    @test H2Factory.nchunks(
        executionscheduling(OhMyThreads.DynamicScheduler(), lv; nchunks=3)
    ) == 3
end

@testset "map innerscheduler defaults to serial for every kernel" begin
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    A = fakegalerkinmap(tree; T=ComplexF64, i2o=scalartranslator(1.0 + 0.0im))
    for op in
        (I2IOp(), I2OOp(), O2OOp(), ModulationOp(), ReverseModulationOp(), LeafMulOp())
        @test H2Factory.innerscheduler(A, op, 3) isa OhMyThreads.SerialScheduler
        # the compatibility alias resolves to the same thing
        @test H2Factory.operationscheduler(A, op, 3) isa OhMyThreads.SerialScheduler
    end
end

@testset "map outerscheduler defaults to the single scheduler (uniform)" begin
    # A map with no outer table serves its single `scheduler(A)` for every phase/level, in any
    # orientation, with only one scheduler stored, the plan distinction cannot matter.
    outer = OhMyThreads.DynamicScheduler()
    A = SchedulerOnlyMap(zeros(ComplexF64, 4, 4), outer)
    for op in (AggregateOp(), I2OOp(), DisaggregateOp())
        @test H2Factory.outerscheduler(A, op, 3) === H2Factory.scheduler(A) === outer
        for M in (transpose(A), adjoint(A))
            @test H2Factory.outerscheduler(M, op, 3) === outer
        end
    end
    # ...and every inner kernel is serial, the production default for a map storing nothing more.
    for op in
        (I2IOp(), I2OOp(), O2OOp(), ModulationOp(), ReverseModulationOp(), LeafMulOp())
        @test H2Factory.innerscheduler(A, op, 3) isa OhMyThreads.SerialScheduler
    end
end

@testset "outer/inner scheduler accessors resolve through transpose/adjoint" begin
    # Regression (this cost 9 errors across the iFMM suite). Both accessors resolve via
    # `forwardmap`/`maptrait`, so a concrete map's override MUST be written at the trait level. An
    # override typed on the bare concrete type is silently skipped the moment the map is wrapped in
    # `transpose(...)`/`adjoint(...)`; the generic fallback fires instead and reaches for
    # `A.scheduler`, which a map storing an `ExecutionScheduling` does not have, a `FieldError`
    # halfway through a matvec. It went unnoticed because nothing asked a *wrapped* map for a
    # scheduler, and the fakes' `scheduler` field would have masked it anyway.
    lv = 1:5
    outer = OhMyThreads.DynamicScheduler()
    A = SchedulingOnlyMap(
        zeros(ComplexF64, 4, 4), defaultexecutionscheduling(lv; outer=outer)
    )

    # forward resolves (baseline)
    @test H2Factory.outerscheduler(A, AggregateOp(), 3) === outer
    # ...and so must both wrappers: scheduling does not flip with orientation (the phase tags name the
    # phase in every orientation; it is the *plan* each phase walks that flips), so every wrapper
    # serves the same stored object rather than a mirrored one.
    for M in (transpose(A), adjoint(A))
        for op in (AggregateOp(), I2OOp(), DisaggregateOp())
            @test H2Factory.outerscheduler(M, op, 3) === outer
        end
        for op in
            (I2IOp(), I2OOp(), O2OOp(), ModulationOp(), ReverseModulationOp(), LeafMulOp())
            @test H2Factory.innerscheduler(M, op, 3) isa OhMyThreads.SerialScheduler
        end
    end

    # The fallback really is a FieldError for this map shape, i.e. the assertions above pass because
    # the override is reached, not because a `scheduler` field quietly answered for it.
    @test_throws Exception H2Factory.scheduler(A)
end

@testset "user-settable nchunks: buffers+schedule stay consistent, matvec unchanged" begin
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    # dense: reference at default nchunks, then every nchunks (below, at, above nthreads) must agree
    Aref = fakegalerkinmap(tree; T=ComplexF64, i2o=scalartranslator(1.5 - 0.5im))
    x = randn(ComplexF64, size(Aref, 2))
    yref = H2Factory.farmul!(similar(x), Aref, x)
    for nc in (1, 2, 3, 16, 32)
        A = fakegalerkinmap(
            tree; T=ComplexF64, i2o=scalartranslator(1.5 - 0.5im), nchunks=nc
        )
        @test H2Factory.aggregationschedule(A).nchunks == nc          # schedule sized to nc
        @test H2Factory.farmul!(similar(x), A, x) ≈ yref             # buffers sized to nc too -> correct
    end
    # Toeplitz path (FFT-plan pool + Toeplitz moments), a missed buffer site would OOB or corrupt
    Tref = faketoeplitzmap(tree, z -> exp(-abs(z[1])); rank=2)
    xt = randn(ComplexF64, size(Tref, 2))
    ytref = H2Factory.farmul!(similar(xt), Tref, xt)
    for nc in (1, 2, 16)
        A = faketoeplitzmap(tree, z -> exp(-abs(z[1])); rank=2, nchunks=nc)
        @test H2Factory.aggregationschedule(A).nchunks == nc
        @test H2Factory.farmul!(similar(xt), A, xt) ≈ ytref
    end
end

end # module TestSchedules
