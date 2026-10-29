module H2Factory
using LinearAlgebra
using LinearMaps
using StaticArrays
using SparseArrays
using ProgressMeter
using ChunkSplitters
import Base.Threads.@threads
using Base.ScopedValues: ScopedValue, with
using OhMyThreads
using H2Trees

import H2Trees: AggregatePlan, DisaggregateTranslatePlan
import H2Trees: AggregateTranslatePlan, DisaggregatePlan
import H2Trees: AbstractAggregationPlan, AbstractDisaggregationPlan

macro memorybuffer()
    memoryexpr = if VERSION >= v"1.11"
        quote
            Memory
        end
    else
        quote
            Array
        end
    end

    return memoryexpr
end

abstract type FarMulMode end

function progressbar(workload, verbose; color=:white, kwargs...)
    return Progress(
        workload; barglyphs=BarGlyphs("[=> ]"), color=color, enabled=verbose, kwargs...
    )
end

# Load protocol declarations before their implementations. Import these bindings explicitly below;
# `using .Interfaces` would allow a later same-named definition to create a separate generic.
include("Interfaces.jl")
import .Interfaces:
    buildmoments,
    buildvectorbuffers,
    buildi2ichannels,
    buildo2ochannels,
    buildi2ochannel,
    schedulinglevels,
    nearcriterion,
    translatingnodesiterator,
    aggregatenode,
    aggregationmode,
    buildnearinteractionsassembler,
    isgalerkinsymmetric,
    storetrait,
    buildstorer,
    applyleafmul!,
    applyconjugatedleafmul!,
    conjmul!,
    leafmul,
    leafmulscheduler,
    coupling,
    modulation,
    applycouplingkernel!,
    modulate_in!,
    modulate_out!,
    i2ochannels,
    borrowbuffer,
    takefftplan!,
    takeifftplan!,
    assertnosymmetry,
    aggregateleafchunk!,
    assemble

include("maps/traits.jl")
include("maps/translationcontext.jl")
include("maps/leveltranslator.jl")
include("maps/operations.jl")
include("maps/scheduling.jl")
include("maps/interactionstrategy.jl")
include("maps/abstracth2map.jl")
include("maps/core.jl")
include("maps/bufferpool.jl")
include("maps/functors.jl")
include("maps/accessors.jl")
include("maps/bundles.jl")
include("maps/schedules.jl")
# Workspace methods need the accessors, schedules, and scheduling types defined above.
include("maps/workspace.jl")

include("moments/abstractmoment.jl")
include("moments/create/createmoment.jl")
include("moments/create/createmomentcollection.jl")
include("moments/create/linkmomenttocollection.jl")
include("moments/create/createlinked.jl")
include("moments/toeplitz/toeplitzmoment.jl")
include("moments/toeplitz/toeplitzmul.jl")
include("moments/store.jl")
include("moments/allocatemoments.jl")
include("moments/aliasing.jl")

@createlinked Moment MomentCollection
@createlinkedtoeplitz ToeplitzMoment ToeplitzMomentCollection

# type aliases:
#   Moment1D{T} = Moment{1}
#   Moment2D{T} = Moment{2}
#   Moment3D{T} = Moment{3}
#   Moment4D{T} = Moment{4}

#   MomentCollection1D{T} = MomentCollection{1}
#   MomentCollection2D{T} = MomentCollection{2}
#   MomentCollection3D{T} = MomentCollection{3}
#   MomentCollection4D{T} = MomentCollection{4}
for D in 1:4
    eval(quote
        $(Symbol("Moment", D, "D")){T} = Moment{$D,T}
        $(Symbol("MomentCollection", D, "D")){T} = MomentCollection{$D,T}
    end)
    eval(
        quote
            $(Symbol("ToeplitzMoment", D, "D")){T} = ToeplitzMoment{$D,T}
            $(Symbol("ToeplitzMomentCollection", D, "D")){T} = ToeplitzMomentCollection{
                $D,T
            }
        end,
    )
end

include("ext/extension_definitions.jl")

include("nearinteractions/abstractnearinteractions.jl")
include("nearinteractions/passthroughnearinteractions.jl")
include("nearinteractions/dummynearinteractions.jl")
include("nearinteractions/randomnearinteractions.jl")
include("nearinteractions/sparsenearinteractions.jl")
include("nearinteractions/blocknearinteractions.jl")
include("nearinteractions/nearmul.jl")
include("nearinteractions/strategies.jl")

include("farinteractions/farmul.jl")
include("farinteractions/traversalhelpers.jl")
include("farinteractions/aggregation.jl")
include("farinteractions/disaggregation.jl")
include("farinteractions/i2otranslations/i2otranslationtraits.jl")
include("farinteractions/i2otranslations/i2oprotocol.jl")
include("farinteractions/i2otranslations/i2otranslationcollection.jl")
include("farinteractions/i2otranslations/symmetryreducedcollection.jl")
include("farinteractions/i2otranslations/symmetryreduceddense.jl")
include("farinteractions/i2otranslations/dummytranslationcollection.jl")
include("farinteractions/i2otranslations/toeplitzi2ocollection.jl")

# test/Package/publicapi.jl records API stability and audience, including supported
# extension hooks that are accessed through qualified names rather than exports.
export GalerkinH2Map, PetrovGalerkinH2Map
# Export workspace constructors, but keep accessors and builder hooks qualified.
export AbstractH2Core, GalerkinH2Core, PetrovGalerkinH2Core
export AbstractH2Workspace, GalerkinH2Workspace, PetrovGalerkinH2Workspace
export H2Workspace, similarworkspace, WorkspaceBuffers, SidedChannels
# Custom top-level operations can use these guards to take exclusive workspace access.
export acquireworkspace!, releaseworkspace!, withworkspace, ConcurrentWorkspaceError
export LevelTranslator, SectorSelector, nsectors, transferkey, leveltransfer
export I2ITranslationCollection, DenseI2I, SeparableI2I, i2itrait
export leveltranslation!, applytransfer!
export LevelTranslators, SidedLevelTranslators, Moments, MomentCollections, VectorBuffers
export Schedules, PlanSchedule, checkexecutiondata
export OuterScheduling, InnerScheduling, ExecutionScheduling
export outerscheduling, outerscheduler, innerscheduler, operationscheduler
export perlevel, serialinnerscheduling
export defaultexecutionscheduling, serialexecutionscheduling, executionscheduling
export AggregateOp,
    I2OOp, DisaggregateOp, I2IOp, O2OOp, ModulationOp, ReverseModulationOp, LeafMulOp
export TrialAggregationOp, TestAggregationOp, TestDisaggregationOp, TrialDisaggregationOp
export nchunks, withnchunks, withschedulechunks, schedulechunkcounts, requestedntasks
# Export the aliasing check and report; keep layout and finding helpers internal.
export checkmomentaliasing, MomentAliasingReport
export @createlinked, @createmoment, @createmomentcollection, @linkmomenttocollection
export @createlinkedtoeplitz, @createtoeplitzmoment
export AbstractMoment, AbstractMomentCollection
export subset
export add!
export storetrait, buildstorer
export StoreAsIs

export AbstractNearInteractionsAssembler
export PassThroughNearInteractionsAssembler
export DummyNearInteractionsAssembler
export RandomNearInteractionsAssembler
export SparseBEASTNearInteractionsAssembler # requires BEAST to work
export BlockBEASTNearInteractionsAssembler # requires BEAST and BlockSparseMatrices to work

export AbstractInteractionStrategy, InteractionStrategy
export nearcriterion, translatingnodesiterator, aggregatenode, aggregationmode
export AbstractNearInteractionsStrategy,
    BlockBEASTNearInteractionsStrategy,
    DummyNearInteractionsStrategy,
    PassThroughNearInteractionsStrategy
export buildnearinteractionsassembler, assemblenearinteractions, assembleplans

export I2OTranslationCollection, DummyI2OTranslationCollection

# Export reduction constructors, result types, and component policies.
# Permutation helpers and lazy translation views remain qualified implementation details.
export SymmetryReducedI2OTranslationCollection, symmetryreducedcollection
export symmetryreducedtoeplitzcollection, symmetryreduceddensecollection
export VectorField, TrueTensor, PseudoTensor

export I2ITranslatorFunctor, I2OTranslatorFunctor, O2OTranslatorFunctor
export AbstractI2OTranslationTrait,
    DiagonalTranslation, DenseTranslation, ToeplitzI2OTranslationTrait

export elementwisemul!
export nearmul, nearmul!, nearinteractions
export farmul, farmul!, farinteractions

# Coupling hooks stay qualified. Collections extend applycouplingkernel!, not the
# guarded applycoupling! entry. Use H2Factory.applyi2o! for a complete per-pair application.
export operator

# H2Trees is not re-exported. The following assembly hooks load with the H2BEAST extension.
export assemblemoments!
export superspace
export assemblymap
export @efie, @mw3Dhypersingular, @mw3Dweaklysingular, @mw3Ddoublelayer
export @hh3Ddoublelayer, @hh3Ddoublelayer_transposed

# Load diagnostics after the types and accessors it inspects. Entry points stay under
# H2Factory.Diagnostics rather than joining the multiplication API.
include("diagnostics/Diagnostics.jl")
export Diagnostics

end # module H2Factory
