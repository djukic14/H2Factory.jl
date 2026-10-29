# Execution scheduling has two independent axes: outer traversal loops are keyed by concrete plans,
# while work inside local kernels is keyed by operation. `ExecutionScheduling` also owns `nchunks`,
# which determines both node partitioning and scratch-pool size.

# --- per-level containers -------------------------------------------------------------------------

"""
    perlevel(scheduler, levels)

Map each tree level in `levels` to `scheduler`.

A typed `Dict` supports noncontiguous level ranges while preserving type-stable lookup.
"""
perlevel(scheduler::S, levels) where {S} =
    Dict{Int,S}(level => scheduler for level in levels)

# --- outer (traversal) scheduling -----------------------------------------------------------------

"""
    OuterScheduling(trialaggregation, testaggregation, i2o, testdisaggregation, trialdisaggregation)

Outer scheduler per traversal plan, stored as per-level containers. Read with
`outerscheduler(outer, op, maptrait, level)`.

Keyed by the concrete plan (= node set) each chunk loop walks, not by the phase, because the four
(phase, orientation) combinations are four *distinct* node sets:

| phase            | forward              | transpose / adjoint   |
|:---------------- |:-------------------- |:--------------------- |
| `AggregateOp`    | `trialaggregation`   | `testaggregation`     |
| `DisaggregateOp` | `testdisaggregation` | `trialdisaggregation` |

`i2o` has no orientation partner and serves both directions. Each field may cover a different level
range, allowing the two sides of a block tree to differ.
"""
struct OuterScheduling{TrA,TeA,I2O,TeD,TrD}
    trialaggregation::TrA
    testaggregation::TeA
    i2o::I2O
    testdisaggregation::TeD
    trialdisaggregation::TrD
end

# Resolve a phase and orientation to the plan it walks.
function outerscheduler(o::OuterScheduling, ::AggregateOp, ::isForward, level::Int)
    return o.trialaggregation[level]
end
function outerscheduler(
    o::OuterScheduling, ::AggregateOp, ::Union{isTranspose,isAdjoint}, level::Int
)
    return o.testaggregation[level]
end
function outerscheduler(o::OuterScheduling, ::DisaggregateOp, ::isForward, level::Int)
    return o.testdisaggregation[level]
end
function outerscheduler(
    o::OuterScheduling, ::DisaggregateOp, ::Union{isTranspose,isAdjoint}, level::Int
)
    return o.trialdisaggregation[level]
end
outerscheduler(o::OuterScheduling, ::I2OOp, ::AbstractMapTrait, level::Int) = o.i2o[level]

# Plan tags address fields directly, without map orientation.
outerscheduler(o::OuterScheduling, ::TrialAggregationOp, level::Int) =
    o.trialaggregation[level]
outerscheduler(o::OuterScheduling, ::TestAggregationOp, level::Int) =
    o.testaggregation[level]
function outerscheduler(o::OuterScheduling, ::TestDisaggregationOp, level::Int)
    return o.testdisaggregation[level]
end
function outerscheduler(o::OuterScheduling, ::TrialDisaggregationOp, level::Int)
    return o.trialdisaggregation[level]
end

# A phase tag without an orientation means forward. Traversal code instead uses the map-level accessor,
# which resolves orientation from the map.
function outerscheduler(o::OuterScheduling, op::AbstractH2Operation, level::Int)
    return outerscheduler(o, op, isForward(), level)
end

"""
    outerscheduling(default, levels)
    outerscheduling(default, levels, (op, level) => scheduler, ...)

Build an [`OuterScheduling`](@ref) with `default` at every plan and level, then apply sparse
`(plan, level) => scheduler` overrides:

    outerscheduling(DynamicScheduler(), levels,
                    (I2OOp(), 3) => DynamicScheduler(; ntasks=8),
                    (TestDisaggregationOp(), 5) => SerialScheduler())

Overrides are stored directly in the per-level containers.
"""
function outerscheduling(default, levels, pairs::Pair...)
    trialagg = _perleveldict(default, levels)
    testagg = _perleveldict(default, levels)
    i2o = _perleveldict(default, levels)
    testdisagg = _perleveldict(default, levels)
    trialdisagg = _perleveldict(default, levels)
    for ((op, level), sched) in pairs
        _setplan!(trialagg, testagg, i2o, testdisagg, trialdisagg, op, level, sched)
    end
    return OuterScheduling(trialagg, testagg, i2o, testdisagg, trialdisagg)
end

# Overrides may contain heterogeneous scheduler types.
_perleveldict(default, levels) = Dict{Int,Any}(level => default for level in levels)

function _setplan!(
    trialagg, testagg, i2o, testdisagg, trialdisagg, ::TrialAggregationOp, level, sched
)
    return (trialagg[level] = sched)
end
function _setplan!(
    trialagg, testagg, i2o, testdisagg, trialdisagg, ::TestAggregationOp, level, sched
)
    return (testagg[level] = sched)
end
function _setplan!(trialagg, testagg, i2o, testdisagg, trialdisagg, ::I2OOp, level, sched)
    return (i2o[level] = sched)
end
function _setplan!(
    trialagg, testagg, i2o, testdisagg, trialdisagg, ::TestDisaggregationOp, level, sched
)
    return (testdisagg[level] = sched)
end
function _setplan!(
    trialagg, testagg, i2o, testdisagg, trialdisagg, ::TrialDisaggregationOp, level, sched
)
    return (trialdisagg[level] = sched)
end

# --- inner (local-kernel) scheduling --------------------------------------------------------------

"""
    InnerScheduling(i2i, i2o, o2o, modulation, reversemodulation, leafmul)

Inner scheduler per local kernel, stored as per-level containers and read with
`innerscheduler(inner, op, level)`.

The production default is serial for every inner kernel. Individual kernels, including batched leaf
multiplication, may be configured independently.
"""
struct InnerScheduling{I2I,I2O,O2O,Mod,RevMod,Leaf}
    i2i::I2I
    i2o::I2O
    o2o::O2O
    modulation::Mod
    reversemodulation::RevMod
    leafmul::Leaf
end

innerscheduler(inner::InnerScheduling, ::I2IOp, level::Int) = inner.i2i[level]
innerscheduler(inner::InnerScheduling, ::I2OOp, level::Int) = inner.i2o[level]
innerscheduler(inner::InnerScheduling, ::O2OOp, level::Int) = inner.o2o[level]
innerscheduler(inner::InnerScheduling, ::ModulationOp, level::Int) = inner.modulation[level]
function innerscheduler(inner::InnerScheduling, ::ReverseModulationOp, level::Int)
    return inner.reversemodulation[level]
end
innerscheduler(inner::InnerScheduling, ::LeafMulOp, level::Int) = inner.leafmul[level]

"""
    serialinnerscheduling(levels; leafmul=SerialScheduler())

Use serial scheduling for every local kernel at every level. `leafmul` may be overridden separately
for implementations that batch leaf work.
"""
function serialinnerscheduling(levels; leafmul=SerialScheduler())
    serial = SerialScheduler()
    return InnerScheduling(
        perlevel(serial, levels),
        perlevel(serial, levels),
        perlevel(serial, levels),
        perlevel(serial, levels),
        perlevel(serial, levels),
        perlevel(leafmul, levels),
    )
end

# --- the bundle -----------------------------------------------------------------------------------

"""
    ExecutionScheduling(outer, inner, nchunks)

Bundle an [`OuterScheduling`](@ref), an [`InnerScheduling`](@ref), and node-partition granularity
`nchunks`. Map constructors store this and expose it through
`outerscheduler(A, op, level)` / `innerscheduler(A, op, level)` / [`nchunks`](@ref)`(A)`.
Build one with [`defaultexecutionscheduling`](@ref) or [`serialexecutionscheduling`](@ref).

`nchunks` controls both node partitioning and scratch-pool size. It is independent of worker count,
which is configured by the outer scheduler.
"""
struct ExecutionScheduling{Outer,Inner}
    outer::Outer
    inner::Inner
    nchunks::Int
end

"""
    nchunks(scheduling::ExecutionScheduling)

Return the node-partition granularity and corresponding buffer-pool size.
"""
nchunks(e::ExecutionScheduling) = e.nchunks

"""
    defaultexecutionscheduling(levels; outer=DynamicScheduler(), nchunks=Threads.nthreads(),
                               leafmul=SerialScheduler())

Use `outer` for every traversal plan, serial inner kernels, and one chunk per Julia thread.

`leafmul` overrides the inner scheduler for batched leaf work.
"""
function defaultexecutionscheduling(
    levels;
    outer=DynamicScheduler(),
    nchunks::Int=Threads.nthreads(),
    leafmul=SerialScheduler(),
)
    return ExecutionScheduling(
        OuterScheduling(
            perlevel(outer, levels),
            perlevel(outer, levels),
            perlevel(outer, levels),
            perlevel(outer, levels),
            perlevel(outer, levels),
        ),
        serialinnerscheduling(levels; leafmul=leafmul),
        nchunks,
    )
end

"""
    serialexecutionscheduling(levels; nchunks=1)

Use serial outer and inner scheduling. The default is one chunk.
"""
function serialexecutionscheduling(levels; nchunks::Int=1)
    serial = SerialScheduler()
    return ExecutionScheduling(
        OuterScheduling(
            perlevel(serial, levels),
            perlevel(serial, levels),
            perlevel(serial, levels),
            perlevel(serial, levels),
            perlevel(serial, levels),
        ),
        serialinnerscheduling(levels),
        nchunks,
    )
end

"""
    requestedntasks(scheduler)

Return the scheduler's requested task count, used as its natural chunk and scratch-pool count.

`SerialScheduler` returns one. Schedulers without a fixed or discoverable count fall back to
`Threads.nthreads()`.
"""
requestedntasks(::SerialScheduler) = 1

function requestedntasks(scheduler)
    n = _ohmythreadsnchunks(scheduler)
    return n isa Integer && n > 0 ? Int(n) : Threads.nthreads()
end

function _ohmythreadsnchunks(scheduler)
    isdefined(OhMyThreads.Schedulers, :nchunks) || return nothing
    return try
        OhMyThreads.Schedulers.nchunks(scheduler)
    catch
        nothing
    end
end

# Convert a bare scheduler to the default outer/inner configuration. Chunk count defaults to the
# scheduler's requested task count and may be overridden independently.
function executionscheduling(
    s, levels; nchunks::Int=requestedntasks(s), leafmul=SerialScheduler()
)
    return defaultexecutionscheduling(levels; outer=s, nchunks=nchunks, leafmul=leafmul)
end
# Preserve a prebuilt schedule. `leafmul` is accepted so callers can pass it to either overload.
executionscheduling(e::ExecutionScheduling, levels; leafmul=nothing) = e
