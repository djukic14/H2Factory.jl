# A core holds data shared by all multiplications. Each workspace holds its own temporary data,
# so several workspaces can use the same core at once.

"""
    AbstractH2Core{T}

Base type for the shared, read-only data of an H2 map. `T` is the element type.

Core types provide read-only accessors that take `(core, maptrait)`. They also provide the methods
needed to build a workspace: [`schedulinglevels`](@ref), [`plans`](@ref), [`buildmoments`](@ref),
[`buildvectorbuffers`](@ref), [`buildi2ichannels`](@ref), [`buildo2ochannels`](@ref), and
[`buildi2ochannel`](@ref).

Make a subtype of [`GalerkinH2Core`](@ref) or [`PetrovGalerkinH2Core`](@ref). To multiply with a
core, first build an [`H2Workspace`](@ref) from it.
"""
abstract type AbstractH2Core{T} end

"""
    GalerkinH2Core{T} <: AbstractH2Core{T}

Core whose test and trial sides use the same space and tree. [`H2Workspace`](@ref) builds a
`GalerkinH2Workspace` from it.
"""
abstract type GalerkinH2Core{T} <: AbstractH2Core{T} end

"""
    PetrovGalerkinH2Core{T} <: AbstractH2Core{T}

Core whose test and trial sides can use different spaces or trees. [`H2Workspace`](@ref) builds a
`PetrovGalerkinH2Workspace` from it.
"""
abstract type PetrovGalerkinH2Core{T} <: AbstractH2Core{T} end

Base.eltype(::AbstractH2Core{T}) where {T} = T

"""
    core(A)

Return the shared [`AbstractH2Core`](@ref) used by a workspace or map.

`transpose(A)` and `adjoint(A)` use the same core as `A`. Passing a core returns that core.
"""
core(A::AbstractH2Core) = A
core(A) = core(forwardmap(A), isForward())
core(A, ::isForward) = A.core

"""
    plans(core)

Return the core's plans. [`H2Workspace`](@ref) uses them to build its own [`Schedules`](@ref).
By default, this reads `core.plans`.
"""
plans(core::AbstractH2Core) = core.plans
