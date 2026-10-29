module TestWorkspace

using Test
using H2Factory
using H2Trees
using OhMyThreads: SerialScheduler, DynamicScheduler, GreedyScheduler

# Unit coverage for the core/workspace abstractions themselves -- `AbstractH2Core`, `H2Workspace`,
# and the six builder hooks -- driven by a toy core rather than a real FMM. The point is to pin the
# CONTRACT that every concrete core (iFMM's two, and any future one) is written against:
#
#   * every hook is called exactly once per workspace, inside ONE `withschedulechunks` scope keyed by
#     that workspace's own schedules;
#   * read-only accessors resolve to the core, in all three orientations;
#   * mutable accessors resolve to the workspace, so two workspaces over one core share nothing;
#   * `nchunks`/`Schedules` belong to the workspace, so two workspaces may be scheduled differently.
#
# Deliberately NOT a matvec test: iFMM's `test/iFMM/workspaces.jl` covers execution over real
# translators, including exact-equality concurrency checks. Here a real traversal would only obscure
# which half of the split a failure came from.

# A plan stand-in exposing just what `Schedules(plans; nchunks)` reads. `planschedule` dispatches on
# the concrete `H2Trees` plan types, so the toy core below carries real plans built over a real tree
# -- the cheapest tree that produces all four.
function toyplans()
    points = [H2Trees.SVector(0.1 * i, 0.2 * i, 0.3 * i) for i in 1:200]
    tree = H2Trees.buildtree(points; builder=H2Trees.TwoNTreeBuilder(; minvalues=10))
    return tree,
    H2Trees.buildplans(tree; builder=H2Trees.PlanBuilder(; isnear=H2Trees.isnear()))
end

# Records what `H2Workspace` asked for, so the test can assert on the CALL PATTERN and not just the
# result: each hook appends the per-level chunk counts visible at the moment it ran.
mutable struct BuildLog
    calls::Vector{Pair{Symbol,Dict{Int,Int}}}
end
BuildLog() = BuildLog(Pair{Symbol,Dict{Int,Int}}[])

struct ToyCore{T,Tree,Plans} <: H2Factory.GalerkinH2Core{T}
    tree::Tree
    plans::Plans
    # The levels some plan actually traverses -- the only levels a buffer pool may be built for.
    # `nchunks(level)` indexes STRICTLY and raises for anything else (a pool for an untraversed level
    # is a sizing bug, not a case to paper over), so a builder looping over the tree's full level
    # range instead would `KeyError` on the root. Exactly the discipline the real cores follow.
    bufferlevels::Vector{Int}
    log::BuildLog
    # Distinguishable stand-ins for the read-only data, so an accessor resolving to the wrong half is
    # visible as a wrong VALUE rather than only as a missing method.
    marker::Symbol
end

function ToyCore(::Type{T}) where {T}
    tree, plans = toyplans()
    bufferlevels = sort!(
        collect(keys(H2Factory.schedulechunkcounts(H2Factory.Schedules(plans; nchunks=1))))
    )
    return ToyCore{T,typeof(tree),typeof(plans)}(
        tree, plans, bufferlevels, BuildLog(), :core
    )
end

# The scheduling containers span every tree level, a superset of the traversed ones.
H2Factory.schedulinglevels(c::ToyCore) = H2Trees.levels(c.tree)

# Read-only accessors, at the same `(object, maptrait)` arity a real core implements.
H2Trees.tree(c::ToyCore, ::H2Factory.isForward) = c.tree
H2Trees.isnear(c::ToyCore, ::H2Factory.isForward) = c.marker
H2Factory.nearinteractions(c::ToyCore, ::H2Factory.isForward) = zeros(ComplexF64, 4, 4)
H2Factory.farmulmode(c::ToyCore, ::H2Factory.isForward) = c.marker
H2Factory.minrelevantlevel(c::ToyCore, ::H2Factory.isForward) = first(c.bufferlevels)
H2Factory.testmomentcollections(c::ToyCore, ::H2Factory.isForward) = :testcollections
H2Factory.trialmomentcollections(c::ToyCore, ::H2Factory.isForward) = :trialcollections
H2Factory.testaggregationplan(c::ToyCore, ::H2Factory.isForward) =
    c.plans.testaggregationplan
H2Factory.trialaggregationplan(c::ToyCore, ::H2Factory.isForward) =
    c.plans.trialaggregationplan
H2Factory.testdisaggregationplan(c::ToyCore, ::H2Factory.isForward) =
    c.plans.testdisaggregationplan
H2Factory.trialdisaggregationplan(c::ToyCore, ::H2Factory.isForward) =
    c.plans.trialdisaggregationplan

# The orientation-covering group (see `maps/workspace.jl`): these have an `::Any`-trait generic body,
# so all three traits must be answered or a transposed read resolves to a `FieldError`.
for trait in (:isForward, :isTranspose, :isAdjoint)
    @eval H2Factory.operator(c::ToyCore, ::H2Factory.$trait) = :operator
    @eval H2Factory.verbose(c::ToyCore, ::H2Factory.$trait) = false
    @eval H2Factory.i2otranslator(c::ToyCore, ::H2Factory.$trait) = :i2o
end
H2Factory.α(c::ToyCore, ::H2Factory.isForward) = 1
H2Factory.α(c::ToyCore, ::H2Factory.isTranspose) = 2
H2Factory.α(c::ToyCore, ::H2Factory.isAdjoint) = 3

H2Factory.triali2itranslator(c::ToyCore) = :triali2i
H2Factory.testi2itranslator(c::ToyCore) = :testi2i
H2Factory.trialo2otranslator(c::ToyCore) = :trialo2o
H2Factory.testo2otranslator(c::ToyCore) = :testo2o

# The six builder hooks. Each records the chunk counts in scope when it ran and returns a FRESH
# object, so "two workspaces share nothing" is checkable by identity.
function _record!(c::ToyCore, name::Symbol)
    counts = Dict(level => H2Factory.nchunks(level) for level in c.bufferlevels)
    push!(c.log.calls, name => counts)
    return counts
end

function H2Factory.buildmoments(c::ToyCore)
    _record!(c, :moments)
    return [0], H2Factory.Moments(Dict{Int,Int}(), Dict{Int,Int}())
end
H2Factory.buildvectorbuffers(c::ToyCore) = (_record!(c, :vectorbuffers); Dict{Int,Int}())
H2Factory.buildi2ichannels(c::ToyCore) = (_record!(c, :i2ichannels); Dict{Int,Int}())
H2Factory.buildo2ochannels(c::ToyCore) = (_record!(c, :o2ochannels); Dict{Int,Int}())
H2Factory.buildi2ochannel(c::ToyCore) = (_record!(c, :i2ochannel); Dict{Int,Int}())
# The only hook with a DEFAULT, so it is overridden here for the same reason as the rest: to prove
# `H2Workspace` calls it, once, inside the same chunk scope. A core that wants no leaf scratch just
# omits it -- covered separately below.
H2Factory.buildleafchannels(c::ToyCore) = (_record!(c, :leafchannels); Dict{Int,Int}())

# A core that leaves `buildleafchannels` alone, to pin the default half of the contract: every
# existing core keeps working and gets the no-op stand-in.
struct LeaflessCore{T,C} <: H2Factory.GalerkinH2Core{T}
    inner::C
end
LeaflessCore(inner::ToyCore{T}) where {T} = LeaflessCore{T,typeof(inner)}(inner)

# Forward the whole contract to the wrapped `ToyCore`; only `buildleafchannels` is left unimplemented.
H2Factory.plans(c::LeaflessCore) = H2Factory.plans(c.inner)
H2Factory.schedulinglevels(c::LeaflessCore) = H2Factory.schedulinglevels(c.inner)
H2Factory.buildmoments(c::LeaflessCore) = H2Factory.buildmoments(c.inner)
H2Factory.buildvectorbuffers(c::LeaflessCore) = H2Factory.buildvectorbuffers(c.inner)
H2Factory.buildi2ichannels(c::LeaflessCore) = H2Factory.buildi2ichannels(c.inner)
H2Factory.buildo2ochannels(c::LeaflessCore) = H2Factory.buildo2ochannels(c.inner)
H2Factory.buildi2ochannel(c::LeaflessCore) = H2Factory.buildi2ochannel(c.inner)

# A map that owns no workspace at all -- the shape the ownership guard must leave alone.
struct FlagLessMap{T} <: H2Factory.GalerkinH2Map{T}
    nearinteractions::Matrix{T}   # only `size(A)` reads it
end

@testset verbose = true "Core/workspace abstractions" begin
    @testset "H2Workspace calls every builder once, inside the chunk scope" begin
        core = ToyCore(ComplexF64)
        ws = H2Workspace(
            core,
            H2Factory.defaultexecutionscheduling(
                H2Factory.schedulinglevels(core); nchunks=3
            ),
        )

        @test ws isa H2Factory.GalerkinH2Workspace
        @test Set(first.(core.log.calls)) == Set([
            :moments, :vectorbuffers, :i2ichannels, :o2ochannels, :i2ochannel, :leafchannels
        ])
        @test length(core.log.calls) == 6

        # THE SCOPE INVARIANT. Every builder must see the SAME counts, and those counts must be the
        # ones this workspace's schedules actually chunk each level into -- not the requested
        # `nchunks` (a level with fewer nodes than requested gets fewer chunks). A pool sized by
        # anything else is an out-of-bounds at `buffer[chunkid]`.
        expected = H2Factory.schedulechunkcounts(H2Factory.schedules(ws))
        for (name, counts) in core.log.calls
            @test counts == Dict(level => expected[level] for level in core.bufferlevels)
        end
    end

    @testset "a bare scheduler is expanded over schedulinglevels(core)" begin
        core = ToyCore(ComplexF64)
        ws = H2Workspace(core, SerialScheduler())
        @test H2Factory.scheduling(ws) isa H2Factory.ExecutionScheduling
        @test H2Factory.outerscheduler(
            ws, H2Factory.AggregateOp(), first(core.bufferlevels)
        ) isa SerialScheduler
    end

    @testset "a bare scheduler sizes the pools by what it asks for" begin
        # The pools follow the SCHEDULER'S intent, not the machine's thread count -- otherwise a
        # serial or deliberately capped workspace still allocates one buffer per Julia thread, and
        # the "one 4-task workspace, one serial workspace" story is not true of the buffers.
        core = ToyCore(ComplexF64)

        # Serial: one chunk, matching `serialexecutionscheduling`'s own default. These two spellings
        # of the same intent used to disagree.
        @test H2Factory.nchunks(H2Workspace(core, SerialScheduler())) == 1
        @test H2Factory.nchunks(H2Workspace(core, SerialScheduler())) ==
            H2Factory.nchunks(H2Workspace(core, H2Factory.serialexecutionscheduling(1:1)))

        # A capped scheduler: its own cap, independent of `Threads.nthreads()`.
        @test H2Factory.nchunks(H2Workspace(core, DynamicScheduler(; ntasks=3))) == 3
        # A bare dynamic scheduler still means one chunk per thread, as before.
        @test H2Factory.nchunks(H2Workspace(core, DynamicScheduler())) == Threads.nthreads()

        # ...and the explicit override wins over both.
        @test H2Factory.nchunks(H2Workspace(core, SerialScheduler(); nchunks=5)) == 5
        @test H2Factory.nchunks(
            H2Workspace(core, DynamicScheduler(; ntasks=3); nchunks=7)
        ) == 7

        # Combining `nchunks` with a prebuilt scheduling is two contradictory statements of the same
        # dial, and the loser would silently be the buffer-pool size. Refused.
        prebuilt = H2Factory.defaultexecutionscheduling(
            H2Factory.schedulinglevels(core); nchunks=2
        )
        @test H2Factory.nchunks(H2Workspace(core, prebuilt)) == 2
        @test_throws ArgumentError H2Workspace(core, prebuilt; nchunks=4)
    end

    @testset "requestedntasks reports what a scheduler asks for" begin
        @test H2Factory.requestedntasks(SerialScheduler()) == 1
        @test H2Factory.requestedntasks(DynamicScheduler(; ntasks=6)) == 6
        @test H2Factory.requestedntasks(DynamicScheduler()) == Threads.nthreads()
        # Anything that exposes no count at all falls back to the thread count -- over-allocating
        # scratch is wasteful, never wrong.
        @test H2Factory.requestedntasks(GreedyScheduler()) == Threads.nthreads()
    end

    @testset "checkexecutiondata judges against the map's OWN scheduler" begin
        core = ToyCore(ComplexF64)
        # The intended concurrent configuration: a 3-task workspace with 3 chunks. Silent, even
        # though `nchunks < Threads.nthreads()` may well hold -- sibling workspaces use the rest of
        # the machine, and this one cannot know about them.
        @test_logs H2Factory.checkexecutiondata(
            H2Workspace(core, DynamicScheduler(; ntasks=3))
        )
        @test_logs H2Factory.checkexecutiondata(H2Workspace(core, SerialScheduler()))
    end

    @testset "read-only accessors resolve to the core, in every orientation" begin
        core = ToyCore(ComplexF64)
        ws = H2Workspace(core, SerialScheduler())

        @test H2Factory.core(ws) === core
        @test H2Factory.core(transpose(ws)) === core
        @test H2Factory.core(adjoint(ws)) === core
        @test_throws MethodError H2Factory.core(ws, H2Factory.isTranspose())
        @test_throws MethodError H2Factory.core(ws, H2Factory.isAdjoint())
        # A core is its own core, which is what keeps the forwarding total.
        @test H2Factory.core(core) === core

        @test H2Trees.tree(ws) === core.tree
        @test H2Trees.tree(transpose(ws)) === core.tree
        @test H2Trees.isnear(ws) === :core
        @test H2Factory.farmulmode(ws) === :core
        @test H2Factory.minrelevantlevel(ws) == first(core.bufferlevels)
        @test H2Factory.operator(ws) === :operator
        @test H2Factory.i2otranslator(ws) === :i2o
        @test H2Factory.i2otranslator(transpose(ws)) === :i2o
        @test H2Factory.i2otranslator(adjoint(ws)) === :i2o

        # `α` is the one that genuinely differs per orientation; a forwarding bug that collapsed the
        # three traits into one would be invisible on every other accessor here.
        @test H2Factory.α(ws) == 1
        @test H2Factory.α(transpose(ws)) == 2
        @test H2Factory.α(adjoint(ws)) == 3

        # A Galerkin workspace's two sides are its one tree, inherited from `GalerkinH2Map`.
        @test H2Trees.testtree(ws) === core.tree
        @test H2Trees.trialtree(ws) === core.tree

        # Plans flip test<->trial under transpose/adjoint, resolved by the generic accessors against
        # the core's forward answers.
        @test H2Factory.aggregationplan(ws) === core.plans.trialaggregationplan
        @test H2Factory.aggregationplan(transpose(ws)) === core.plans.testaggregationplan
        @test H2Factory.disaggregationplan(ws) === core.plans.testdisaggregationplan
        @test H2Factory.disaggregationplan(transpose(ws)) ===
            core.plans.trialdisaggregationplan
    end

    @testset "mutable accessors resolve to the workspace" begin
        core = ToyCore(ComplexF64)
        a = H2Workspace(core, SerialScheduler())
        b = H2Workspace(core, SerialScheduler())

        @test H2Factory.core(a) === H2Factory.core(b)
        # Nothing mutable is shared. Checked field by field: `!==` on the bundles alone would pass
        # even if every buffer inside were the same object.
        for field in fieldnames(H2Factory.WorkspaceBuffers)
            @test getfield(H2Factory.workspacebuffers(a), field) !==
                getfield(H2Factory.workspacebuffers(b), field)
        end
        @test H2Factory.testmoments(a) !== H2Factory.testmoments(b)
        @test H2Factory.momentbuffers(a) !== H2Factory.momentbuffers(b)
        @test H2Factory.i2ochannel(a) !== H2Factory.i2ochannel(b)
        # The leaf channel is scratch like any other, so two workspaces must not share it. Asserted
        # through the accessor as well as through the bundle field above: a resolution bug that read
        # the CORE would make both workspaces answer with one object while the fields still differed.
        @test H2Factory.trialleafchannel(a) !== H2Factory.trialleafchannel(b)
        @test H2Factory.testleafchannel(a) !== H2Factory.testleafchannel(b)

        # Scratch is a property of the workspace, not of the view of it.
        @test H2Factory.workspacebuffers(transpose(a)) === H2Factory.workspacebuffers(a)
        @test H2Factory.i2ochannel(adjoint(a)) === H2Factory.i2ochannel(a)
    end

    @testset "leaf channels resolve by side, and the phase accessors alias them" begin
        core = ToyCore(ComplexF64)
        ws = H2Workspace(core, SerialScheduler())
        stored = H2Factory.workspacebuffers(ws).leafchannels

        # A GALERKIN core has one tree, so its two sides are the same leaves at the same levels and a
        # bare (unsided) channel serves both. Aliasing here is the intent, not an accident -- a second
        # copy would only be a duplicate of the first.
        @test H2Factory.trialleafchannel(ws) === stored
        @test H2Factory.testleafchannel(ws) === stored

        # The phase accessors are aliases over the SIDE accessors, in exactly the sense
        # `aggregationmomentcollection`/`disaggregationmomentcollection` are: aggregation is the trial
        # side's phase, disaggregation the test side's. Pinned by identity with the side accessor
        # rather than with `stored`, so the two layers cannot drift apart while both still "work".
        @test H2Factory.aggregationleafchannel(ws) === H2Factory.trialleafchannel(ws)
        @test H2Factory.disaggregationleafchannel(ws) === H2Factory.testleafchannel(ws)

        # ...and the flip lives in the side accessors, so a transposed/adjoint read of one side
        # resolves to the OTHER side's storage. Unobservable by value on a Galerkin core (both sides
        # alias), so this pins the RESOLUTION PATH: `trialleafchannel(transpose(ws))` must arrive via
        # `testleafchannel(ws)`, which is what makes it correct for a Petrov core, where the two
        # genuinely differ (asserted on the real Petrov fake in `farinteractions/test_leafchannel.jl`).
        for view in (transpose(ws), adjoint(ws))
            @test H2Factory.trialleafchannel(view) === H2Factory.testleafchannel(ws)
            @test H2Factory.testleafchannel(view) === H2Factory.trialleafchannel(ws)
            @test H2Factory.aggregationleafchannel(view) === H2Factory.testleafchannel(ws)
            @test H2Factory.disaggregationleafchannel(view) ===
                H2Factory.trialleafchannel(ws)
        end
    end

    @testset "buildleafchannels defaults to EmptyChannels" begin
        # The one builder hook with a default, so that every core written before the leaf channel
        # existed keeps working untouched -- and gets a stand-in that `borrowbuffer` answers for any
        # level and chunk, rather than a missing-method error at the first leaf multiply.
        core = LeaflessCore(ToyCore(ComplexF64))
        ws = H2Workspace(core, SerialScheduler())

        @test H2Factory.workspacebuffers(ws).leafchannels === H2Factory.EmptyChannels()
        @test H2Factory.trialleafchannel(ws) === H2Factory.EmptyChannels()
        @test H2Factory.aggregationleafchannel(transpose(ws)) === H2Factory.EmptyChannels()
        @test H2Factory.borrowbuffer(
            H2Factory.disaggregationleafchannel(ws), 3, 1, H2Factory.maptrait(ws)
        ) === H2Factory.EmptyChannels()
    end

    @testset "similarworkspace allocates a fresh sibling over the same core" begin
        core = ToyCore(ComplexF64)
        original = H2Workspace(core, DynamicScheduler(; ntasks=3))
        sibling = similarworkspace(original)

        @test sibling isa typeof(original)
        @test H2Factory.core(sibling) === H2Factory.core(original)
        @test H2Factory.scheduling(sibling) === H2Factory.scheduling(original)
        @test H2Factory.nchunks(sibling) == H2Factory.nchunks(original)
        @test H2Factory.schedules(sibling) !== H2Factory.schedules(original)
        @test H2Factory.workspacebuffers(sibling) !== H2Factory.workspacebuffers(original)

        for field in fieldnames(H2Factory.WorkspaceBuffers)
            @test getfield(H2Factory.workspacebuffers(sibling), field) !==
                getfield(H2Factory.workspacebuffers(original), field)
        end

        serial = similarworkspace(original; scheduling=SerialScheduler())
        @test H2Factory.core(serial) === H2Factory.core(original)
        @test H2Factory.nchunks(serial) == 1
        capped = similarworkspace(original; scheduling=DynamicScheduler(), nchunks=2)
        @test H2Factory.nchunks(capped) == 2
        @test_throws ArgumentError similarworkspace(original; nchunks=2)
    end

    @testset "scheduling and schedules belong to the workspace" begin
        core = ToyCore(ComplexF64)
        lv = H2Factory.schedulinglevels(core)
        serial = H2Workspace(core, H2Factory.serialexecutionscheduling(lv))
        chunked = H2Workspace(core, H2Factory.defaultexecutionscheduling(lv; nchunks=4))

        # Two workspaces over ONE core, scheduled differently -- the reason `nchunks` lives here and
        # not on the core.
        @test H2Factory.nchunks(serial) == 1
        @test H2Factory.nchunks(chunked) == 4
        @test H2Factory.schedules(serial) !== H2Factory.schedules(chunked)
        @test H2Factory.aggregationschedule(serial).nchunks == 1
        @test H2Factory.aggregationschedule(chunked).nchunks == 4

        @test H2Factory.scheduling(transpose(serial)) === H2Factory.scheduling(serial)
        @test H2Factory.schedules(adjoint(serial)) === H2Factory.schedules(serial)
    end

    @testset "the guard is a no-op for a map that owns no shared scratch" begin
        # `acquireworkspace!` is called unconditionally by the generic entry points, so it has to be
        # harmless for the lightweight `AbstractH2Map` implementations that keep no workspace.
        A = FlagLessMap(zeros(ComplexF64, 4, 4))
        @test H2Factory.busyflag(A) === nothing
        @test H2Factory.acquireworkspace!(A) === nothing
        @test H2Factory.releaseworkspace!(A) === nothing
        # ...and twice in a row, since there is nothing to contend for.
        @test H2Factory.acquireworkspace!(A) === nothing
        @test H2Factory.withworkspace(() -> 42, A) == 42
    end

    @testset "a core is not executable" begin
        core = ToyCore(ComplexF64)
        # The split's central safety property: only the workspace is a `LinearMap`, so `mul!` cannot
        # compile against a core and then fail deep inside the traversal for want of buffers.
        @test !(core isa H2Factory.LinearMaps.LinearMap)
        @test H2Workspace(core, SerialScheduler()) isa H2Factory.LinearMaps.LinearMap
        @test eltype(core) === ComplexF64
        @test eltype(H2Workspace(core, SerialScheduler())) === ComplexF64
    end
end

end
