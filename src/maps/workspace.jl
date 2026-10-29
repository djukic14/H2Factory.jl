# A workspace owns the mutable state of one matvec. Several workspaces may share a read-only core,
# but each workspace may run only one operation at a time. It is also the executable `LinearMap`.
# Its Galerkin or Petrov type preserves the side-specific accessor dispatch.
# Scheduling lives here because each workspace may use a different number of scratch buffers.

"""
    SidedChannels(test, trial)

Hold separate translator scratch channels for the test and trial sides of a Petrov workspace.
Galerkin workspaces use one channel without this wrapper.
"""
struct SidedChannels{Test,Trial}
    test::Test
    trial::Trial
end

"""
    WorkspaceBuffers(moments, momentbuffers, vectorbuffers, i2ichannels, o2ochannels, i2ochannel,
                     leafchannels)

Hold the moments and scratch used by one workspace. `momentbuffers` keeps the backing memory for
`moments` alive. The channel fields hold transfer and leaf-multiply scratch, including FFT plans
where needed. [`H2Workspace`](@ref) builds the buffers using the workspace's chunk counts.
"""
struct WorkspaceBuffers{M,MB,VB,I2IC,O2OC,I2OC,LC}
    moments::M
    momentbuffers::MB
    vectorbuffers::VB
    i2ichannels::I2IC
    o2ochannels::O2OC
    i2ochannel::I2OC
    leafchannels::LC
end

# The two types have identical fields but different supertypes for side-specific dispatch.
for (Workspace, Map) in (
    (:GalerkinH2Workspace, :GalerkinH2Map),
    (:PetrovGalerkinH2Workspace, :PetrovGalerkinH2Map),
)
    @eval begin
        struct $Workspace{T,C,S,Sch,B} <: $Map{T}
            core::C
            scheduling::S
            schedules::Sch
            buffers::B
            # Atomic ownership flag; the shared core has no such flag.
            busy::Threads.Atomic{Bool}
        end

        # Take the element type from the core.
        function $Workspace(
            core::AbstractH2Core{T}, scheduling, schedules, buffers
        ) where {T}
            return $Workspace{
                T,typeof(core),typeof(scheduling),typeof(schedules),typeof(buffers)
            }(
                core, scheduling, schedules, buffers, Threads.Atomic{Bool}(false)
            )
        end
    end
end

"""
    AbstractH2Workspace{T}

The union of Galerkin and Petrov workspaces with element type `T`. Both are executable H2 maps.
"""
const AbstractH2Workspace{T} = Union{GalerkinH2Workspace{T},PetrovGalerkinH2Workspace{T}}

"""
    H2Workspace(core, scheduling) -> GalerkinH2Workspace | PetrovGalerkinH2Workspace

Allocate an executable workspace over the read-only `core`. Each call creates fresh moments,
buffers, and scratch channels. Workspaces may share a core, but a workspace cannot run two
operations at once.

Pass an [`ExecutionScheduling`](@ref) or a bare scheduler. For a bare scheduler, `nchunks` can
override the number of scratch chunks. A prebuilt `ExecutionScheduling` already sets that count,
so it cannot be combined with `nchunks`.
"""
function H2Workspace(core::AbstractH2Core, scheduling; nchunks::Union{Nothing,Int}=nothing)
    execution = _workspacescheduling(
        scheduling, schedulinglevels(core), nchunks, leafmulscheduler(core)
    )
    # Qualify the accessor because the `nchunks` keyword shadows its name.
    schedules = Schedules(plans(core); nchunks=H2Factory.nchunks(execution))
    buffers = withschedulechunks(schedules) do
        momentbuffers, moments = buildmoments(core)
        return WorkspaceBuffers(
            moments,
            momentbuffers,
            buildvectorbuffers(core),
            buildi2ichannels(core),
            buildo2ochannels(core),
            buildi2ochannel(core),
            buildleafchannels(core),
        )
    end
    return _workspace(core, execution, schedules, buffers)
end

"""
    similarworkspace(ws::AbstractH2Workspace; scheduling=scheduling(ws), nchunks=nothing)

Allocate fresh mutable buffers over `core(ws)`.
Reuse the execution scheduling by default, or pass `scheduling` to change it.
The `nchunks` restriction is the same as for [`H2Workspace`](@ref).
"""
function similarworkspace(
    ws::AbstractH2Workspace; scheduling=scheduling(ws), nchunks::Union{Nothing,Int}=nothing
)
    return H2Workspace(core(ws), scheduling; nchunks=nchunks)
end

# By default, leaf multiplication uses a serial inner scheduler.
leafmulscheduler(::AbstractH2Core) = SerialScheduler()

function _workspacescheduling(scheduling, levels, nchunks::Nothing, leafmul)
    return executionscheduling(scheduling, levels; leafmul=leafmul)
end
function _workspacescheduling(scheduling, levels, nchunks::Int, leafmul)
    return executionscheduling(scheduling, levels; nchunks=nchunks, leafmul=leafmul)
end
function _workspacescheduling(
    scheduling::ExecutionScheduling, levels, nchunks::Int, leafmul
)
    return throw(
        ArgumentError(
            "H2Workspace: `nchunks=$nchunks` was passed alongside a prebuilt ExecutionScheduling, " *
            "which already carries nchunks=$(H2Factory.nchunks(scheduling)). Set it on the " *
            "scheduling (e.g. `defaultexecutionscheduling(levels; nchunks=$nchunks)`), or pass a " *
            "bare scheduler with `nchunks`.",
        ),
    )
end

_workspace(core::GalerkinH2Core, args...) = GalerkinH2Workspace(core, args...)
_workspace(core::PetrovGalerkinH2Core, args...) = PetrovGalerkinH2Workspace(core, args...)

# --- workspace construction hooks --------------------------------------------------------------
# Builders run once per workspace. Use `nchunks(level)` when sizing pooled buffers.

"""
    buildleafchannels(core)

Build leaf-multiply scratch for this workspace. The default [`EmptyChannels`](@ref) needs no
scratch. A core that needs it may return one channel or a [`SidedChannels`](@ref) pair.
Size per-level pools with `nchunks(level)`.
"""
buildleafchannels(core) = EmptyChannels()

# --- workspace accessors ---------------------------------------------------------------------------

"""
    workspacebuffers(A)

Return the [`WorkspaceBuffers`](@ref) behind `A`. Forward, transpose, and adjoint views share them.
"""
workspacebuffers(A) = workspacebuffers(forwardmap(A), maptrait(A))
workspacebuffers(A::AbstractH2Workspace, ::AbstractMapTrait) = A.buffers

"""
    scheduling(A)

Return the workspace's [`ExecutionScheduling`](@ref).
"""
scheduling(A) = scheduling(forwardmap(A), maptrait(A))
scheduling(A::AbstractH2Workspace, ::AbstractMapTrait) = A.scheduling

"""
    schedules(A)

Return the workspace's materialized [`Schedules`](@ref).
"""
schedules(A) = schedules(forwardmap(A), maptrait(A))
schedules(A::AbstractH2Workspace, ::AbstractMapTrait) = A.schedules

nchunks(A::AbstractH2Workspace) = nchunks(A.scheduling)

# Read-only accessors forward to the core. Use concrete traits to avoid ambiguity with the generic
# accessor methods. These accessors handle transpose and adjoint by returning to the forward entry.
for accessor in (
    :(H2Trees.tree),
    :(H2Trees.isnear),
    :nearinteractions,
    :farmulmode,
    :minrelevantlevel,
    :testaggregationplan,
    :trialaggregationplan,
    :testdisaggregationplan,
    :trialdisaggregationplan,
    :testmomentcollections,
    :trialmomentcollections,
)
    @eval $accessor(A::AbstractH2Workspace, trait::isForward) = $accessor(core(A), trait)
end

# These accessors need each orientation. In particular, `α` and `leafmul` may vary by direction.
for accessor in (:operator, :verbose, :α, :i2otranslator, :leafmul)
    @eval begin
        $accessor(A::AbstractH2Workspace, trait::isForward) = $accessor(core(A), trait)
        $accessor(A::AbstractH2Workspace, trait::isTranspose) = $accessor(core(A), trait)
        $accessor(A::AbstractH2Workspace, trait::isAdjoint) = $accessor(core(A), trait)
    end
end

# Side-specific level translators are read-only core data.
for accessor in
    (:triali2itranslator, :testi2itranslator, :trialo2otranslator, :testo2otranslator)
    @eval $accessor(A::AbstractH2Workspace) = $accessor(core(A))
end

# Mutable workspace data.

# The same moment pool is used in every orientation.
momentbuffers(A::AbstractH2Workspace, ::AbstractMapTrait) =
    workspacebuffers(A).momentbuffers

testmoments(A::AbstractH2Workspace, ::isForward) = workspacebuffers(A).moments.test
trialmoments(A::AbstractH2Workspace, ::isForward) = workspacebuffers(A).moments.trial

# Schedule selection follows the plan selected by the map trait.
testaggregationschedule(A::AbstractH2Workspace, ::isForward) = schedules(A).testaggregation
trialaggregationschedule(A::AbstractH2Workspace, ::isForward) =
    schedules(A).trialaggregation
testdisaggregationschedule(A::AbstractH2Workspace, ::isForward) =
    schedules(A).testdisaggregation
trialdisaggregationschedule(A::AbstractH2Workspace, ::isForward) =
    schedules(A).trialdisaggregation

# Keep the trait for outer scheduling: it selects the plan in each orientation.
function outerscheduler(
    A::AbstractH2Workspace, trait::AbstractMapTrait, op::AbstractH2Operation, level::Int
)
    return outerscheduler(scheduling(A).outer, op, trait, level)
end

function innerscheduler(
    A::AbstractH2Workspace, ::AbstractMapTrait, op::AbstractH2Operation, level::Int
)
    return innerscheduler(scheduling(A).inner, op, level)
end

# Galerkin uses one vector-buffer set; Petrov uses separate test and trial sets.
vectorbuffer(A::GalerkinH2Workspace, node) = workspacebuffers(A).vectorbuffers[node]
testvectorbuffer(A::PetrovGalerkinH2Workspace, node) =
    workspacebuffers(A).vectorbuffers.test[node]
trialvectorbuffer(A::PetrovGalerkinH2Workspace, node) =
    workspacebuffers(A).vectorbuffers.trial[node]

# Galerkin shares transfer channels between sides. Petrov keeps each side's channels separate.
triali2ichannel(A::GalerkinH2Workspace) = workspacebuffers(A).i2ichannels
testi2ichannel(A::GalerkinH2Workspace) = workspacebuffers(A).i2ichannels
testo2ochannel(A::GalerkinH2Workspace) = workspacebuffers(A).o2ochannels
trialo2ochannel(A::GalerkinH2Workspace) = workspacebuffers(A).o2ochannels

triali2ichannel(A::PetrovGalerkinH2Workspace) = workspacebuffers(A).i2ichannels.trial
testi2ichannel(A::PetrovGalerkinH2Workspace) = workspacebuffers(A).i2ichannels.test
testo2ochannel(A::PetrovGalerkinH2Workspace) = workspacebuffers(A).o2ochannels.test
trialo2ochannel(A::PetrovGalerkinH2Workspace) = workspacebuffers(A).o2ochannels.trial

# Leaf channels may be sided or shared; the default `EmptyChannels()` is shared.
_leafside(channels::SidedChannels, ::Val{:test}) = channels.test
_leafside(channels::SidedChannels, ::Val{:trial}) = channels.trial
_leafside(channels, ::Val) = channels

testleafchannel(A::AbstractH2Workspace, ::isForward) =
    _leafside(workspacebuffers(A).leafchannels, Val(:test))
trialleafchannel(A::AbstractH2Workspace, ::isForward) =
    _leafside(workspacebuffers(A).leafchannels, Val(:trial))

# Direction is carried by the translator, so I2O uses one channel in every orientation.
i2ochannel(A::AbstractH2Workspace, ::Any) = workspacebuffers(A).i2ochannel

# --- workspace ownership -------------------------------------------------------------------------
# Top-level operations acquire the workspace once; their inner kernels do not reacquire it.
"""
    acquireworkspace!(A)

Take exclusive ownership of the workspace behind `A`, or throw if it is already in use.
Transpose and adjoint views share the same flag. This is a no-op for maps without a workspace.
Pair it with [`releaseworkspace!`](@ref) in a `finally` block.
"""
acquireworkspace!(A) = _acquireworkspace!(busyflag(A), A)

"""
    releaseworkspace!(A)

Release ownership taken by [`acquireworkspace!`](@ref). Idempotent, and a no-op for a non-workspace.
"""
releaseworkspace!(A) = _releaseworkspace!(busyflag(A))

"""
    busyflag(A)

The ownership flag of the workspace behind `A`, or `nothing` if `A` is not workspace-backed.
"""
busyflag(A) = busyflag(forwardmap(A), maptrait(A))
busyflag(A::AbstractH2Workspace, ::AbstractMapTrait) = A.busy
busyflag(A, ::AbstractMapTrait) = nothing

_acquireworkspace!(::Nothing, A) = nothing

function _acquireworkspace!(flag::Threads.Atomic{Bool}, A)
    # CAS prevents two tasks from acquiring the same workspace.
    Threads.atomic_cas!(flag, false, true) && throw(
        ConcurrentWorkspaceError(
            "H2Workspace is already in use; create a separate workspace for concurrent operations " *
            "(`H2Workspace(core, scheduling)` per task, the core is shared and read-only, only " *
            "the workspace's buffers are exclusive).",
        ),
    )
    return nothing
end

_releaseworkspace!(::Nothing) = nothing
_releaseworkspace!(flag::Threads.Atomic{Bool}) = (flag[]=false; nothing)

"""
    ConcurrentWorkspaceError(msg)

Thrown when a second top-level operation tries to use an occupied workspace.
"""
struct ConcurrentWorkspaceError <: Exception
    msg::String
end

Base.showerror(io::IO, e::ConcurrentWorkspaceError) =
    print(io, "ConcurrentWorkspaceError: ", e.msg)

"""
    withworkspace(f, A)

Run `f()` while owning the workspace behind `A`. Release it even if `f` throws.
"""
function withworkspace(f, A)
    acquireworkspace!(A)
    try
        return f()
    finally
        releaseworkspace!(A)
    end
end
