"""
    AbstractAssemblyStrategy

Base type for strategies that configure how kernel matrix entries are assembled into stored
form. The same strategy type serves near and far interactions; each side passes its own
instance.

Subtypes provide [`matrixdata`](@ref), [`storage`](@ref) and `scheduler`. The defaults read
fields with these names, so other layouts override the accessors.
"""
abstract type AbstractAssemblyStrategy end

"""
    AssemblyStrategy(matrixdata, storage; scheduler=DynamicScheduler())
    AssemblyStrategy(matrixdata, storage, scheduler)

Hold the settings for assembling a kernel matrix into stored form.

  - `matrixdata`: passed to [`buildkernelmatrix`](@ref); selects how entries are
    evaluated, e.g. a quadrature strategy.
  - `storage`: selects the stored format of the assembled entries.
  - `scheduler`: the OhMyThreads scheduler distributing the assembly work.
"""
struct AssemblyStrategy{MatrixData,Storage,Scheduler} <: AbstractAssemblyStrategy
    matrixdata::MatrixData
    storage::Storage
    scheduler::Scheduler
end

function AssemblyStrategy(matrixdata, storage; scheduler=DynamicScheduler())
    return AssemblyStrategy(matrixdata, storage, scheduler)
end

"""
    matrixdata(strategy::AbstractAssemblyStrategy)

Return the data selecting how `strategy` evaluates kernel matrix entries.
"""
matrixdata(strategy::AbstractAssemblyStrategy) = strategy.matrixdata

"""
    storage(strategy::AbstractAssemblyStrategy)

Return the stored format `strategy` assembles into.
"""
storage(strategy::AbstractAssemblyStrategy) = strategy.storage

scheduler(strategy::AbstractAssemblyStrategy) = strategy.scheduler
