# Map data is read through accessors so each map type can choose where to store it. The one-argument
# form unwraps `transpose(A)` or `adjoint(A)` and passes the direction to the two-argument form.

"""
    @traitaccessor name forwardbody flipbody

Define the usual three methods for an accessor:

    name(A)                                 = name(forwardmap(A), maptrait(A))
    name(A, ::isForward)                    = forwardbody
    name(A, ::Union{isTranspose,isAdjoint}) = flipbody

`forwardbody` and `flipbody` use `A` and can name fields or other accessors. Use this macro for
accessors whose transpose and adjoint forms both swap test and trial sides.
"""
macro traitaccessor(name, forwardbody, flipbody)
    return esc(quote
        function $name(A::AbstractH2Maps)
            return $name(forwardmap(A), maptrait(A))
        end
        function $name(A, ::isForward)
            return $forwardbody
        end
        function $name(A, ::Union{isTranspose,isAdjoint})
            return $flipbody
        end
    end)
end

# Moment buffers ###########################################################################

"""
    momentbuffers(A)

Return the memory pools backing the test and trial moments. They must stay alive while the moments
use them. The same pools serve every map direction.

Map types with different storage should override `momentbuffers(A, trait)`. Workspaces keep their
own pools in [`WorkspaceBuffers`](@ref).
"""
function momentbuffers(A::AbstractH2Maps)
    return momentbuffers(forwardmap(A), maptrait(A))
end

function momentbuffers(A, ::AbstractMapTrait)
    return A.momentbuffers
end

# Tree #####################################################################################

function H2Trees.tree(A::AbstractH2Maps)
    return H2Trees.tree(forwardmap(A), maptrait(A))
end

function H2Trees.tree(A, ::isForward)
    return A.tree
end

function H2Trees.tree(A, ::Union{isTranspose,isAdjoint})
    return H2Trees.tree(A)
end

function H2Trees.trialtree(A::GalerkinH2Map)
    return H2Trees.tree(A)
end

function H2Trees.testtree(A::GalerkinH2Map)
    return H2Trees.tree(A)
end

function aggregationtree(A)
    return H2Trees.trialtree(A)
end

function H2Trees.testtree(A::AbstractH2Maps)
    return H2Trees.testtree(forwardmap(A), maptrait(A))
end

function H2Trees.testtree(A, ::isForward)
    return H2Trees.testtree(H2Trees.tree(A))
end

function H2Trees.testtree(A, ::Union{isAdjoint,isTranspose})
    return H2Trees.trialtree(A)
end

function H2Trees.trialtree(A::AbstractH2Maps)
    return H2Trees.trialtree(forwardmap(A), maptrait(A))
end

function H2Trees.trialtree(A, ::isForward)
    return H2Trees.trialtree(H2Trees.tree(A))
end

function H2Trees.trialtree(A, ::Union{isTranspose,isAdjoint})
    return H2Trees.testtree(A)
end

function disaggregationtree(A)
    return H2Trees.testtree(A)
end

"""
    blocktreesides(A_H2, tree)

Return the row and column trees of a map backed by a `BlockTree`, in the map's current direction.

If a map accessor returns the whole block tree, read its sides directly.
"""
function blocktreesides(A_H2, tree)
    rowside = H2Trees.testtree(A_H2)
    colside = H2Trees.trialtree(A_H2)
    (rowside === tree || colside === tree) &&
        return (H2Trees.testtree(tree), H2Trees.trialtree(tree))
    return (rowside, colside)
end

# Near / far ###############################################################################

"""
    H2Trees.isnear(A::AbstractH2Maps)

Return the near-interaction predicate stored by `A`.

Transpose and adjoint maps use the same predicate as the forward map.
"""
function H2Trees.isnear(A::AbstractH2Maps)
    return H2Trees.isnear(forwardmap(A), maptrait(A))
end

function H2Trees.isnear(A::AbstractH2Map, ::isForward)
    return A.isnear
end

function H2Trees.isnear(A::AbstractH2Map, ::Union{isTranspose,isAdjoint})
    return H2Trees.isnear(A)
end

function H2Trees.isfar(A::AbstractH2Maps)
    return H2Trees.isfar(H2Trees.isnear(A))
end

function nearinteractions end

function nearinteractions(A)
    return nearinteractions(forwardmap(A), maptrait(A))
end

function nearinteractions(A, ::isForward)
    return A.nearinteractions
end

function nearinteractions(A, ::isTranspose)
    return transpose(nearinteractions(A))
end

function nearinteractions(A, ::isAdjoint)
    return adjoint(nearinteractions(A))
end

# Operator / verbose / α / farmulmode ######################################################

function operator(A::AbstractH2Map)
    return operator(forwardmap(A), maptrait(A))
end

function operator(A, ::Any)
    return A.operator
end

function verbose(A::AbstractH2Maps)
    return verbose(forwardmap(A), maptrait(A))
end

function verbose(A, ::Any)
    return A.verbose
end

function α(A::AbstractH2Maps)
    return α(forwardmap(A), maptrait(A))
end

function α(A, ::Any)
    return A.α
end

@traitaccessor farmulmode A.farmulmode adjoint(farmulmode(A))

# Scheduling ###############################################################################

function scheduler(A::AbstractH2Maps)
    return scheduler(forwardmap(A), maptrait(A))
end

function scheduler(A, ::Any)
    return A.scheduler
end

# The traversal reads outer scheduling through the map, so direction selects the right plan.
# Custom maps override the form that takes `(A, trait, op, level)` and use the trait when selecting
# a plan.
function outerscheduler(A::AbstractH2Maps, op::AbstractH2Operation, level::Int)
    return outerscheduler(forwardmap(A), maptrait(A), op, level)
end

# Maps with one scheduler use it for every outer phase and level.
function outerscheduler(A, ::AbstractMapTrait, ::AbstractH2Operation, level::Int)
    return scheduler(A)
end

# For transpose and adjoint maps, `scheduledop` swaps the I2I/O2O scheduler keys so scheduling
# stays with the same tree. The kernel being run does not change.
function innerscheduler(A::AbstractH2Maps, op::AbstractH2Operation, level::Int)
    trait = maptrait(A)
    return innerscheduler(forwardmap(A), trait, scheduledop(op, trait), level)
end

# Maps without a separate inner schedule run local kernels serially.
function innerscheduler(A, ::AbstractMapTrait, ::AbstractH2Operation, level::Int)
    return SerialScheduler()
end

# Alias for `innerscheduler`.
operationscheduler(A, op::AbstractH2Operation, level::Int) = innerscheduler(A, op, level)

# Aggregation plan #########################################################################

function aggregationplan(A)
    return trialaggregationplan(A)
end

@traitaccessor testaggregationplan A.testaggregationplan trialaggregationplan(A)
@traitaccessor trialaggregationplan A.trialaggregationplan testaggregationplan(A)

# Disaggregation plan ######################################################################

function disaggregationplan(A)
    return testdisaggregationplan(A)
end

@traitaccessor testdisaggregationplan A.testdisaggregationplan trialdisaggregationplan(A)
@traitaccessor trialdisaggregationplan A.trialdisaggregationplan testdisaggregationplan(A)

# Schedule (precomputed chunks) ############################################################
# Chunk schedules follow the same plan and direction as their matching plan accessors. The default
# builds them from the plan; maps may override the accessor to return a stored schedule.

function aggregationschedule(A)
    return trialaggregationschedule(A)
end

@traitaccessor trialaggregationschedule planschedule(trialaggregationplan(A)) testaggregationschedule(
    A
)
@traitaccessor testaggregationschedule planschedule(testaggregationplan(A)) trialaggregationschedule(
    A
)

function disaggregationschedule(A)
    return testdisaggregationschedule(A)
end

@traitaccessor testdisaggregationschedule planschedule(testdisaggregationplan(A)) trialdisaggregationschedule(
    A
)
@traitaccessor trialdisaggregationschedule planschedule(trialdisaggregationplan(A)) testdisaggregationschedule(
    A
)

# Moment ###################################################################################

function aggregationmoment(A)
    return trialmoment(A)
end

@traitaccessor testmoments A.testmoments trialmoments(A)
@traitaccessor trialmoments A.trialmoments testmoments(A)
@traitaccessor testmoment NodeGetter(testmoments, A) trialmoment(A)
@traitaccessor trialmoment NodeGetter(trialmoments, A) testmoment(A)

function disaggregationmoment(A)
    return testmoment(A)
end

# Moment collection ########################################################################

function aggregationmomentcollection(A)
    return trialmomentcollection(A)
end

# The one-argument form accepts maps. Cores provide the two-argument form through a workspace.
function testmomentcollections(A::AbstractH2Maps)
    return testmomentcollections(forwardmap(A), maptrait(A))
end

function testmomentcollections(A::PetrovGalerkinH2Map, ::isForward)
    return A.testmomentcollections
end

function testmomentcollections(A::GalerkinH2Map, ::isForward)
    return A.momentcollections
end

function testmomentcollections(A, ::Union{isTranspose,isAdjoint})
    return trialmomentcollections(A)
end

@traitaccessor testmomentcollection NodeGetter(testmomentcollections, A) trialmomentcollection(
    A
)

# Use the same map-only entry point as `testmomentcollections`.
function trialmomentcollections(A::AbstractH2Maps)
    return trialmomentcollections(forwardmap(A), maptrait(A))
end

function trialmomentcollections(A::H2, ::isForward) where {H2<:PetrovGalerkinH2Map}
    return A.trialmomentcollections
end

function trialmomentcollections(A::H2, ::isForward) where {H2<:GalerkinH2Map}
    return A.momentcollections
end

function trialmomentcollections(A, ::Union{isTranspose,isAdjoint})
    return testmomentcollections(A)
end

@traitaccessor trialmomentcollection NodeGetter(trialmomentcollections, A) testmomentcollection(
    A
)

function disaggregationmomentcollection(A)
    return testmomentcollection(A)
end

# i2i translator ###########################################################################

function i2itranslatorfunctor(A)
    return i2itranslatorfunctor(forwardmap(A), maptrait(A))
end

# Choose the side and direction before building the functor. The resolved translator keeps that
# choice when the functor runs.
function i2itranslatorfunctor(A, trait)
    return I2ITranslatorFunctor(i2itranslator(A, trait), aggregationdirection(trait))
end

# Aggregation uses the trial side forward and the test side after transpose or adjoint. Each side
# keeps its I2I and O2O accessors; disaggregation applies the chosen side's I2I transfer with the
# needed transpose or adjoint. Maps with separate transfers override the per-side accessors below.
triali2itranslator(A) = A.i2itranslator
testi2itranslator(A) = A.i2itranslator
trialo2otranslator(A) = A.o2otranslator
testo2otranslator(A) = A.o2otranslator

function i2itranslator(A::AbstractH2Maps)
    return i2itranslator(forwardmap(A), maptrait(A))
end

i2itranslator(A, ::isForward) = triali2itranslator(A)
i2itranslator(A, ::Union{isTranspose,isAdjoint}) = testi2itranslator(A)

# i2o translator ###########################################################################

function i2otranslatorfunctor(A)
    return i2otranslatorfunctor(forwardmap(A), maptrait(A))
end

function i2otranslatorfunctor(A, ::isForward)
    return I2OTranslatorFunctor(A, ForwardTranslation())
end

# Transpose and adjoint maps swap receiving and translating nodes. Their directions apply
# `transpose(coupling)` and `adjoint(coupling)`, respectively.
function i2otranslatorfunctor(A, ::isTranspose)
    return I2OTranslatorFunctor(A, TransposeTranslation())
end
function i2otranslatorfunctor(A, ::isAdjoint)
    return I2OTranslatorFunctor(A, AdjointTranslation())
end

function i2otranslator(A)
    return i2otranslator(forwardmap(A), maptrait(A))
end

function i2otranslator(A, ::isForward)
    return A.i2otranslator
end

function i2otranslator(A, ::Union{isAdjoint,isTranspose})
    return A.i2otranslator
end

# o2o translator ###########################################################################

function o2otranslatorfunctor(A)
    return o2otranslatorfunctor(forwardmap(A), maptrait(A))
end

function o2otranslatorfunctor(A, trait)
    return O2OTranslatorFunctor(
        disaggregationi2itranslator(A, trait), disaggregationdirection(trait)
    )
end

# Disaggregation uses the test side forward and the trial side after transpose or adjoint. The
# functor applies the needed direction to that side's I2I transfer.
disaggregationi2itranslator(A) = disaggregationi2itranslator(forwardmap(A), maptrait(A))
disaggregationi2itranslator(A, ::isForward) = testi2itranslator(A)
disaggregationi2itranslator(A, ::Union{isTranspose,isAdjoint}) = triali2itranslator(A)

# Forward and transpose maps disaggregate with `transpose(I2I)`; adjoint maps use `adjoint(I2I)`.
disaggregationdirection(::isForward) = TransposeTranslation()
disaggregationdirection(::isTranspose) = TransposeTranslation()
disaggregationdirection(::isAdjoint) = AdjointTranslation()

# Accessors for stored O2O transfers. The traversal above resolves I2I and applies its direction.
function o2otranslator(A::AbstractH2Maps)
    return o2otranslator(forwardmap(A), maptrait(A))
end

o2otranslator(A, ::isForward) = testo2otranslator(A)
o2otranslator(A, ::Union{isTranspose,isAdjoint}) = trialo2otranslator(A)

# Leaf multiplication ######################################################################
# `leafmul` selects the functor for both aggregation and disaggregation leaves. Forward and transpose
# use the plain multiply. Adjoint maps must choose a functor based on whether their leaf collections
# need conjugation.
function leafmul(A)
    return leafmul(forwardmap(A), maptrait(A))
end

# Transpose changes the order of operations but does not conjugate the leaf collections.
leafmul(A, ::isForward) = LinAlgebraMulFunctor()
leafmul(A, ::isTranspose) = LinAlgebraMulFunctor()

# Adjoint multiplication conjugates both leaf collections. `eltype(A)` does not tell us whether
# those collections are real, so each map must choose the correct functor explicitly.
function leafmul(A, ::isAdjoint)
    return error(
        "leafmul is not declared for $(typeof(A)) with isAdjoint(). Define " *
        "`H2Factory.leafmul(::T, ::H2Factory.isAdjoint)` to return " *
        "`H2Factory.LinAlgebraMulFunctor()` for real leaf collections, or a conjugating " *
        "functor for complex ones.",
    )
end

# minrelevantlevel #########################################################################

# The relevant level range is the same in every map direction.
@traitaccessor minrelevantlevel A.minrelevantlevel minrelevantlevel(A)

# Vector buffers ###########################################################################

function aggregationvectorbuffer(A, node)
    return aggregationvectorbuffer(forwardmap(A), maptrait(A), node)
end

function aggregationvectorbuffer(A, ::Union{isAdjoint,isTranspose}, node)
    return disaggregationvectorbuffer(A, node)
end

function aggregationvectorbuffer(A, ::isForward, node)
    return trialvectorbuffer(A, node)
end

function disaggregationvectorbuffer(A, node)
    return disaggregationvectorbuffer(forwardmap(A), maptrait(A), node)
end

function disaggregationvectorbuffer(A, ::Union{isAdjoint,isTranspose}, node)
    return aggregationvectorbuffer(A, node)
end

function disaggregationvectorbuffer(A, ::isForward, node)
    return testvectorbuffer(A, node)
end

function testvectorbuffer(A::PetrovGalerkinH2Map, node)
    return A.testvectorbuffer[node]
end

function testvectorbuffer(A::GalerkinH2Map, node)
    return vectorbuffer(A, node)
end

function trialvectorbuffer(A::PetrovGalerkinH2Map, node)
    return A.trialvectorbuffer[node]
end

function trialvectorbuffer(A::GalerkinH2Map, node)
    return vectorbuffer(A, node)
end

function vectorbuffer(A::GalerkinH2Map, node)
    return A.vectorbuffer[node]
end

# Channels #################################################################################

function i2ochannel(A, ::Any)
    return A.i2ochannel
end

function i2ochannel(A)
    return i2ochannel(forwardmap(A), maptrait(A))
end

# Scratch follows the selected tree side and transfer type. A transpose swaps test and trial sides;
# aggregation still uses I2I scratch and disaggregation still uses O2O scratch.
triali2ichannel(A) = A.i2ichannel
testi2ichannel(A) = A.i2ichannel
testo2ochannel(A) = A.o2ochannel
trialo2ochannel(A) = A.o2ochannel

i2ichannel(A, ::isForward) = triali2ichannel(A)
i2ichannel(A, ::Union{isTranspose,isAdjoint}) = testi2ichannel(A)

o2ochannel(A, ::isForward) = testo2ochannel(A)
o2ochannel(A, ::Union{isTranspose,isAdjoint}) = trialo2ochannel(A)

function i2ichannel(A)
    return i2ichannel(forwardmap(A), maptrait(A))
end

function o2ochannel(A)
    return o2ochannel(forwardmap(A), maptrait(A))
end

# Leaf scratch belongs to the same side as its moment collection. Transpose and adjoint swap test
# and trial sides. The phase accessors below name the side used by each traversal phase.
function trialleafchannel(A)
    return trialleafchannel(forwardmap(A), maptrait(A))
end

trialleafchannel(A, ::isForward) = A.leafchannel
trialleafchannel(A, ::Union{isTranspose,isAdjoint}) = testleafchannel(A)

function testleafchannel(A)
    return testleafchannel(forwardmap(A), maptrait(A))
end

testleafchannel(A, ::isForward) = A.leafchannel
testleafchannel(A, ::Union{isTranspose,isAdjoint}) = trialleafchannel(A)

# Aggregation uses trial-side scratch; disaggregation uses test-side scratch.
aggregationleafchannel(A) = trialleafchannel(A)
disaggregationleafchannel(A) = testleafchannel(A)
