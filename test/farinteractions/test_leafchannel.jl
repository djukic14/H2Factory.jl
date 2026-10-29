module TestLeafChannel

using Test
using Random
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils

# THE LEAF-MULTIPLY SCRATCH CHANNEL, end to end through a real traversal.
#
# `buildleafchannels` gives a workspace a per-`(level, chunk)` intermediate that a leaf moment
# collection's `mul!` may stage through -- the shape MLFMA's lazy receive collection needs, where
# mirroring the incoming moment into a real far field and handing BLAS one contiguous matvec beats
# running a fused elementwise kernel. Nothing in H2Factory itself uses the buffer (`applyleafmul!`
# defaults to a plain `LinearAlgebra.mul!` and discards it), so without this file the whole channel
# would be built, borrowed, threaded through four traversal kernels and never once observed.
#
# These tests are deliberately STRUCTURAL, for the reason spelled out at length in
# `test_sideownership.jl`: the numbers are identical whether or not the buffer arrives, comes from the
# right level, or comes from the right SIDE. Only identity can express those.
#
# The contract under test (H2Factory `maps/accessors.jl` + `maps/workspace.jl`):
#
#   * stored and resolved BY SIDE (`trialleafchannel`/`testleafchannel`), borrowed BY PHASE
#     (`aggregationleafchannel` = trial's phase, `disaggregationleafchannel` = test's), with the
#     trial<->test flip living in the side accessors -- the moment-collection model exactly, because
#     this scratch stages for a leaf moment collection and the two must never resolve differently;
#   * keyed by the LEAF's own level, with NO `±1` offset (unlike the level-transfer channels, which
#     are keyed by the child level the transfer spans);
#   * one private buffer per chunk, so two concurrently running chunks never share one.

# --- the probe --------------------------------------------------------------------------------
#
# Records what each leaf multiply was handed. `exclusive` is the concurrency check: a call stamps the
# whole buffer with a tag unique to itself, yields (giving any task that shares the buffer a chance to
# overwrite it), then re-reads. Nodes within ONE chunk run sequentially, so a same-chunk neighbour
# cannot trip this; only a buffer genuinely shared between concurrent chunks can.
struct LeafProbe
    lock::ReentrantLock
    counter::Threads.Atomic{Int}
    buffers::Vector{Any}
    lengths::Vector{Int}
    exclusive::Vector{Bool}
    phases::Vector{Symbol}
end

LeafProbe() =
    LeafProbe(ReentrantLock(), Threads.Atomic{Int}(0), Any[], Int[], Bool[], Symbol[])

struct ProbeCollection{C}
    inner::C
    probe::LeafProbe
end

function _record!(probe::LeafProbe, buffer::Vector{T}, phase::Symbol) where {T}
    tag = T(Threads.atomic_add!(probe.counter, 1) + 1)
    fill!(buffer, tag)
    yield()
    exclusive = all(==(tag), buffer)
    lock(probe.lock) do
        push!(probe.buffers, buffer)
        push!(probe.lengths, length(buffer))
        push!(probe.exclusive, exclusive)
        return push!(probe.phases, phase)
    end
    return nothing
end

# The two `applyleafmul!` arities: 4-arg is the aggregation leaf multiply (moment <- collection * x),
# 6-arg the disaggregation one (y <- collection' * moment). Both forward to the wrapped collection, so
# the probe is numerically transparent and the traversal's own results stay comparable.
function H2Factory.applyleafmul!(buffer::Vector, moment, collection::ProbeCollection, x)
    _record!(collection.probe, buffer, :aggregate)
    return LinearAlgebra.mul!(moment, collection.inner, x)
end

function H2Factory.applyleafmul!(
    buffer::Vector, y, collection::ProbeCollection, moment, α, β
)
    _record!(collection.probe, buffer, :disaggregate)
    return LinearAlgebra.mul!(y, collection.inner, moment, α, β)
end

# Wrap a core's leaf collections in place. The fakes hold them in a `Dict{Int,Any}`, so this needs no
# cooperation from the fake and leaves every other part of the core untouched.
function wrapcollections!(collections::AbstractDict, probe::LeafProbe)
    for (leaf, mc) in collect(collections)
        collections[leaf] = ProbeCollection(mc, probe)
    end
    return collections
end

const RANK = 2

function galerkintree()
    points = [SVector(0.1 * i, 0.2 * i, 0.3 * i) for i in 1:200]
    return H2Trees.buildtree(points; builder=H2Trees.TwoNTreeBuilder(; minvalues=10))
end

# `fakeleafscratchlength(rank, level) = rank + level`, so a buffer's LENGTH says which level's pool it
# came from -- which is what turns "was the borrow keyed correctly?" into an assertion.
levelofbuffer(len::Int) = len - RANK

poolbuffers(channels, level) = channels[level].buffers

# Is `buffer` one of the buffers `channels` holds for `level`? By identity, never by value: two pools
# can hold equal-looking vectors, and identity is the whole question here.
inpool(channels, level, buffer) = any(b -> b === buffer, poolbuffers(channels, level))

@testset verbose = true "Leaf-multiply scratch channel" begin
    @testset "the leaf multiply is handed its chunk's buffer, keyed by the leaf's own level" begin
        tree = galerkintree()
        probe = LeafProbe()
        core = fakegalerkincore(tree; rank=RANK)
        wrapcollections!(
            H2Factory.trialmomentcollections(core, H2Factory.isForward()), probe
        )
        ws = H2Workspace(core, SerialScheduler())

        x = randn(MersenneTwister(1), ComplexF64, H2Trees.numberofvalues(tree))
        y = similar(x)
        mul!(y, ws, x)

        # Both leaf multiplies ran -- otherwise "the buffer is correct" would be vacuous for one of the
        # two phases, and the aggregation site is the one a Galerkin map reaches only via its trial
        # collections while disaggregation reaches the same dict as the test side.
        @test !isempty(probe.phases)
        @test :aggregate in probe.phases
        @test :disaggregate in probe.phases

        # Every borrowed buffer came from the leaf's OWN level, out of this workspace's own pool. A
        # `level ± 1` borrow -- which is what the level-transfer channels correctly do, and therefore
        # the mistake to guard against -- lands on a different level's pool and fails here.
        channels = H2Factory.trialleafchannel(ws)
        leaflevels = sort(unique(H2Trees.level(tree, l) for l in H2Trees.leaves(tree)))
        for (buffer, len) in zip(probe.buffers, probe.lengths)
            level = levelofbuffer(len)
            @test level in leaflevels
            @test inpool(channels, level, buffer)
        end

        # ...and the pool really is level-dependent in shape, so the check above has something to
        # catch. (A single-level tree would make it vacuous.)
        @test length(leaflevels) > 1
        @test length(unique(probe.lengths)) == length(leaflevels)
    end

    @testset "the probe is numerically transparent" begin
        # Guards the tests above: if wrapping the collections changed the result, every structural
        # assertion here would be about a traversal that no longer computes the right thing.
        tree = galerkintree()
        plain = H2Workspace(fakegalerkincore(tree; rank=RANK), SerialScheduler())

        probed = fakegalerkincore(tree; rank=RANK)
        wrapcollections!(
            H2Factory.trialmomentcollections(probed, H2Factory.isForward()), LeafProbe()
        )
        wrapped = H2Workspace(probed, SerialScheduler())

        x = randn(MersenneTwister(2), ComplexF64, H2Trees.numberofvalues(tree))
        a = similar(x)
        b = similar(x)
        @test mul!(a, plain, x) == mul!(b, wrapped, x)
    end

    @testset "concurrent chunks never share a leaf buffer" begin
        # The buffers are per `(level, chunk)` and the traversal indexes them by `chunkid`, so this is
        # the property that makes the whole channel safe under the threaded traversal. Needs a real
        # parallel scheduler AND more than one chunk; with `nchunks=1` it is trivially true.
        tree = galerkintree()
        probe = LeafProbe()
        core = fakegalerkincore(tree; rank=RANK)
        wrapcollections!(
            H2Factory.trialmomentcollections(core, H2Factory.isForward()), probe
        )
        ws = H2Workspace(core, DynamicScheduler(); nchunks=4)

        x = randn(MersenneTwister(3), ComplexF64, H2Trees.numberofvalues(tree))
        mul!(similar(x), ws, x)

        @test !isempty(probe.exclusive)
        @test all(probe.exclusive)
        # More than one distinct buffer was actually handed out, so the check above is not passing
        # merely because everything ran on one chunk.
        @test length(unique(objectid.(probe.buffers))) > 1
    end

    @testset "nchunks sizes the per-level leaf pools" begin
        tree = galerkintree()
        core = fakegalerkincore(tree; rank=RANK)
        for nchunks in (1, 3)
            ws = H2Workspace(core, DynamicScheduler(); nchunks=nchunks)
            channels = H2Factory.trialleafchannel(ws)
            counts = H2Factory.schedulechunkcounts(H2Factory.schedules(ws))
            for level in keys(channels)
                # EXACTLY the level's own chunk count -- not the requested `nchunks` (a level with
                # fewer nodes gets fewer chunks) and not a max over levels. An over-allocated pool
                # would let a mis-keyed `chunkid` read a spare buffer instead of failing.
                @test length(poolbuffers(channels, level)) == counts[level]
                @test all(
                    b -> length(b) == fakeleafscratchlength(RANK, level),
                    poolbuffers(channels, level),
                )
            end
        end
    end

    @testset "leaf pools exist for exactly the traversed levels that hold leaves" begin
        # THE ALLOCATION CONTRACT, stated as an equality rather than two one-sided checks, so one
        # failure message says which way it broke.
        #
        # Neither direction is a size preference. Too MANY: both leaf multiplies fire inside an
        # `H2Trees.isleaf(tree, node)` branch, so a pool at an internal-only level can never be
        # borrowed -- it is dead memory, and it hands a mis-keyed borrow a plausible-looking buffer
        # where `EmptyChannels()` would have been the honest answer. Too FEW: a leaf level with no
        # pool silently borrows `EmptyChannels()`, and a collection that wanted the scratch falls
        # back to whatever its no-scratch path is, with no error and no wrong number to notice.
        #
        # The intersection, not the leaf levels alone: a leaf at a level no plan traverses is never
        # aggregated or disaggregated (`nchunks(level)` has no count for it either), so it correctly
        # gets nothing. The fakes are held to the same rule as MLFMA's `leafscratchchannel`, since a
        # test utility more permissive than the production allocator is a weaker net.
        tree = galerkintree()
        ws = H2Workspace(fakegalerkincore(tree; rank=RANK), DynamicScheduler(); nchunks=2)
        channels = H2Factory.trialleafchannel(ws)
        leaflevels = Set(H2Trees.level(tree, leaf) for leaf in H2Trees.leaves(tree))
        traversed = Set(keys(H2Factory.schedulechunkcounts(H2Factory.schedules(ws))))

        @test Set(keys(channels)) == intersect(traversed, leaflevels)
        # Non-vacuity, both ways: the fixture must actually HAVE pools, and must actually have
        # traversed levels with no leaves -- otherwise the equality above is satisfied by an empty
        # set, or by the filter being a no-op.
        @test !isempty(channels)
        @test !isempty(setdiff(traversed, leaflevels))
    end

    @testset "buildleafchannels runs once per workspace, and per workspace only" begin
        tree = galerkintree()
        leafcalls = Ref(0)
        empties() = H2Factory.EmptyChannels()
        core = fakegalerkincore(
            tree;
            rank=RANK,
            scratch=FakeScratch(
                empties, empties, empties, () -> (leafcalls[] += 1; empties())
            ),
        )

        @test leafcalls[] == 0            # the CORE builds no scratch; the hook is a workspace hook
        H2Workspace(core, SerialScheduler())
        @test leafcalls[] == 1
        H2Workspace(core, SerialScheduler())
        @test leafcalls[] == 2
    end

    @testset "two workspaces over one core get distinct leaf channels" begin
        tree = galerkintree()
        core = fakegalerkincore(tree; rank=RANK)
        a = H2Workspace(core, SerialScheduler())
        b = H2Workspace(core, SerialScheduler())

        @test H2Factory.trialleafchannel(a) !== H2Factory.trialleafchannel(b)
        # ...down to the buffers, not just the enclosing dicts: two dicts holding the same
        # `BufferStore`s would still race.
        for level in keys(H2Factory.trialleafchannel(a))
            for buffer in poolbuffers(H2Factory.trialleafchannel(a), level)
                @test !inpool(H2Factory.trialleafchannel(b), level, buffer)
            end
        end
    end

    @testset "a Galerkin core's two sides alias, deliberately" begin
        tree = galerkintree()
        ws = H2Workspace(fakegalerkincore(tree; rank=RANK), SerialScheduler())
        # ONE tree, so the two sides are the same leaves at the same levels and the same sampling. A
        # second copy would only duplicate the first, so `buildleafchannels` returns a bare channel
        # rather than a `SidedChannels` and both side accessors answer with it.
        @test H2Factory.trialleafchannel(ws) === H2Factory.testleafchannel(ws)
        @test H2Factory.aggregationleafchannel(ws) ===
            H2Factory.disaggregationleafchannel(ws)
    end

    @testset "a Petrov core's two sides are separate, and the phases reach the right one" begin
        # The case a Galerkin map cannot express: two trees, so the sides' leaves sit at different
        # levels with differently shaped scratch, and a phase reaching for the wrong side is a real
        # defect rather than a harmless alias.
        testpoints = [(0.1 * i, 0.2 * i, 0.3 * i) for i in 1:120]
        trialpoints = [(0.1 * i + 5.0, 0.2 * i, 0.3 * i) for i in 1:120]
        probe = LeafProbe()
        core = fakepetrovcore(
            testpoints, trialpoints; rank=RANK, testminvalues=4, trialminvalues=4
        )
        wrapcollections!(
            H2Factory.testmomentcollections(core, H2Factory.isForward()), probe
        )
        wrapcollections!(
            H2Factory.trialmomentcollections(core, H2Factory.isForward()), probe
        )
        ws = H2Workspace(core, SerialScheduler())

        trialchannels = H2Factory.trialleafchannel(ws)
        testchannels = H2Factory.testleafchannel(ws)
        @test trialchannels !== testchannels

        # Phase resolution, forward and flipped. Asserted on the ACCESSORS first, so a failure says
        # which layer is wrong before the traversal check below says that something is.
        @test H2Factory.aggregationleafchannel(ws) === trialchannels
        @test H2Factory.disaggregationleafchannel(ws) === testchannels
        for view in (transpose(ws), adjoint(ws))
            @test H2Factory.aggregationleafchannel(view) === testchannels
            @test H2Factory.disaggregationleafchannel(view) === trialchannels
        end

        # ...and now through a real traversal: every aggregation leaf multiply must have been handed a
        # buffer out of the TRIAL pool and every disaggregation one out of the TEST pool. This is the
        # assertion a numerical check cannot make -- both pools hold plausible-looking vectors, and
        # the traversal's output is identical either way.
        trialtree = H2Trees.trialtree(core.tree)
        testtree = H2Trees.testtree(core.tree)
        x = randn(MersenneTwister(4), ComplexF64, H2Trees.numberofvalues(trialtree))
        y = zeros(ComplexF64, H2Trees.numberofvalues(testtree))
        mul!(y, ws, x)

        @test :aggregate in probe.phases
        @test :disaggregate in probe.phases
        for (buffer, len, phase) in zip(probe.buffers, probe.lengths, probe.phases)
            level = levelofbuffer(len)
            expected = phase === :aggregate ? trialchannels : testchannels
            other = phase === :aggregate ? testchannels : trialchannels
            @test inpool(expected, level, buffer)
            @test !inpool(other, level, buffer)
        end

        # The transposed traversal swaps which side each phase reaches for, so the same assertion runs
        # with the roles exchanged. Without this, a resolution that ignored the orientation entirely
        # would pass everything above.
        empty!(probe.buffers)
        empty!(probe.lengths)
        empty!(probe.exclusive)
        empty!(probe.phases)
        mul!(x, transpose(ws), y)

        @test :aggregate in probe.phases
        @test :disaggregate in probe.phases
        for (buffer, len, phase) in zip(probe.buffers, probe.lengths, probe.phases)
            level = levelofbuffer(len)
            expected = phase === :aggregate ? testchannels : trialchannels
            other = phase === :aggregate ? trialchannels : testchannels
            @test inpool(expected, level, buffer)
            @test !inpool(other, level, buffer)
        end
    end
end

end # module TestLeafChannel
