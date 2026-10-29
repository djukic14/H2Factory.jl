"""
    H2Factory.Interfaces

Generic declarations for H2Factory's extension protocol; no methods or types.
H2Factory imports these bindings, so `H2Factory.f` and `H2Factory.Interfaces.f` are
the same function. Membership here does not imply export.
"""
module Interfaces

# --- Workspace construction ----------------------------------------------------------------------
#
# A core is shared while each concurrent matvec receives its own workspace.

"""
    buildmoments(core) -> (momentbuffers, Moments(test, trial))

Allocate test/trial node moments and their backing pools. Return the pools first:
the wrapped moments do not keep their memory alive. See [`H2Factory.Moments`](@ref).
"""
function buildmoments end

"""
    buildvectorbuffers(core)

Allocate this workspace's per-node leaf multiplication scratch.
"""
function buildvectorbuffers end

"""
    buildi2ichannels(core)

Allocate this workspace's I2I (aggregation) level-transfer scratch.
"""
function buildi2ichannels end

"""
    buildo2ochannels(core)

Allocate this workspace's O2O (disaggregation) level-transfer scratch. See
[`buildi2ichannels`](@ref).
"""
function buildo2ochannels end

"""
    buildi2ochannel(core)

Allocate this workspace's I2O scratch, including any matvec-time plans and modulation buffers.

FFT plans must be created per workspace because they own mutable scratch.
"""
function buildi2ochannel end

"""
    schedulinglevels(core)

The level range spanned by the core's scheduling containers.

Concrete cores must define this because the range depends on their tree geometry.
"""
function schedulinglevels end

# --- Near-field strategy -------------------------------------------------------------------------
#
# Accessors let strategy subtypes choose their own storage layout.

"""
    nearcriterion(strategy::AbstractInteractionStrategy)

Return the functor that classifies near and far interactions.
"""
function nearcriterion end

"""
    translatingnodesiterator(strategy::AbstractInteractionStrategy)

Return `strategy`'s translating-node iterator.
"""
function translatingnodesiterator end

"""
    aggregatenode(strategy::AbstractInteractionStrategy)

Return `strategy`'s aggregate-node predicate.
"""
function aggregatenode end

"""
    aggregationmode(strategy::AbstractInteractionStrategy)

Return `strategy`'s aggregation mode.
"""
function aggregationmode end

"""
    buildnearinteractionsassembler(operator, testspace, trialspace, γ, strategy)

Build the near-interaction assembler a strategy selects.

Implement this for each `AbstractNearInteractionsStrategy` subtype to return its assembler.
"""
function buildnearinteractionsassembler end

"""
    isgalerkinsymmetric(operator) -> Bool

Return whether a Galerkin near block is symmetric, allowing one triangle to be mirrored.
The value form calls `isgalerkinsymmetric(typeof(operator))`. Defaults to `false`:
a false symmetry claim would omit required assembly.
"""
function isgalerkinsymmetric end

# --- Moment storage ------------------------------------------------------------------------------
#
# How assembled moment values reach their destination.

"""
    storetrait(op) -> Type{<:AbstractStorer}

The storer type used when assembling moments for `op`.

The value form delegates to `storetrait(typeof(op))`. [`buildstorer`](@ref) constructs the returned
type with its trailing arguments.
"""
function storetrait end

"""
    buildstorer(F, op, storeinfo) -> storer
    buildstorer(Fs, op, storeinfo, tree) -> storer

Build the storer that writes assembled moments for `op` into `F`/`Fs`.

Resolve `op`'s [`storetrait`](@ref) and return a callable `(value, cluster, m, n)` storer.

The four-argument form supports storers that need the tree to map global indices into leaf boxes.
"""
function buildstorer end

# --- Leaf multiplication -------------------------------------------------------------------------
#
# The bottom of the tree, where a moment collection meets the vector.

"""
    applyleafmul!(leafbuffer, moment, momentcollection, x)
    applyleafmul!(leafbuffer, y, momentcollection, moment, α, β)

Multiply a leaf collection with this chunk's private scratch.
The default calls `LinearAlgebra.mul!` and ignores `leafbuffer`.
Specialize for collections needing scratch; buffer and collection always belong to the same side.
"""
function applyleafmul! end

"""
    applyconjugatedleafmul!(leafbuffer, moment, momentcollection, x)
    applyconjugatedleafmul!(leafbuffer, result, momentcollection, moment, α, β)

The conjugated counterpart of [`applyleafmul!`](@ref), with the same buffer contract.
"""
function applyconjugatedleafmul! end

"""
    conjmul!(moment, momentcollection, y)
    conjmul!(moment, momentcollection, y, α, β)
    conjmul!(y, momentcollection, moment[, α, β])

Multiply with conjugation in place. The moment-output forms conjugate the input vector;
the vector-output forms conjugate the transposed product. See each method for its formula.

`@linkmomenttocollection` generates methods for linked pairs only. Other pairs raise
`MethodError`. Use concrete moment and collection types when querying `hasmethod`.
"""
function conjmul! end

"""
    leafmul(A) -> functor

Return the leaf multiplication functor for `A`.
Specialize on `(core, maptrait)`; the map form resolves these through `forwardmap` and `maptrait`.
Forward and transpose have defaults. Adjoint must opt in: use the plain functor only
when both leaf collections are real; otherwise provide conjugation.
"""
function leafmul end

"""
    leafmulscheduler(core) -> scheduler

Return the scheduler for [`H2Factory.LeafMulOp`](@ref) when constructing execution scheduling from
a bare scheduler. The default is `SerialScheduler()`. A prebuilt `ExecutionScheduling` is preserved.
"""
function leafmulscheduler end

# --- I2O composition, modulation and scratch -----------------------------------------------------
#
# Coupling collections expose their components, storage traits, kernels, and scratch requirements.

"""
    coupling(translation)

Return the coupling of an `I2OTranslation`. Bare `AbstractI2OCouplingCollection`s
return themselves. Wrappers must expose their collection or use
`I2OTranslation(coupling, modulation)`.
"""
function coupling end

"""
    modulation(translation)

Return the modulation component of an `I2OTranslation`; see [`coupling`](@ref).

Most `AbstractI2OCouplingCollection` types default to `NoModulation()`. Collections that require a
conversion must provide their policy explicitly or use `I2OTranslation(coupling, modulation)`.
"""
function modulation end

"""
    applycouplingkernel!(out, in, coupling, i2otrait, receivingnode, translatingnode, direction,
                         accumulation, context)

Apply one coupling block after symmetry validation and trait resolution.
Collections implement this kernel, usually dispatching on `i2otrait`.
Use `H2Factory.couplingkey` for the directed key and `H2Factory.directedoperation`
for the matrix orientation.

Extend this kernel, not the guarded `H2Factory.applycoupling!` entry.
Also define [`assertnosymmetry`](@ref) and `i2otrait`; do not repeat the guard here.
"""
function applycouplingkernel! end

"""
    modulate_in!(moment, level, maptrait, modulation, coupling, channel, scheduler, buffer)

Convert a node's aggregation moment into the coupling basis, in place, before the I2O phase.

This runs once per node rather than once per interaction. Implementations may use the coupling,
channel, and borrowed buffer for plans or scratch.
"""
function modulate_in! end

"""
    modulate_out!(moment, level, maptrait, modulation, coupling, channel, scheduler, buffer)

Convert a node's moment back from the coupling basis after I2O.
This is the output stage paired with [`modulate_in!`](@ref).
"""
function modulate_out! end

"""
    i2ochannels(translation, translatingplan)

Allocate per-`(level, chunk)` I2O scratch for a coupling or modulation policy.

Each workspace must own its mutable plans and scratch. `I2OTranslation` delegates to its coupling,
and collections needing no scratch return `EmptyChannels`. Call this inside `withschedulechunks`
when sizing depends on `nchunks(level)`.
"""
function i2ochannels end

"""
    borrowbuffer(channel, chunkid) -> buffer

Return one chunk's private scratch from a channel.

Concurrent chunks must never receive the same mutable buffer. `EmptyChannels` returns a stand-in for
operations that need no scratch.
"""
function borrowbuffer end

"""
    takefftplan!(channel, level, chunkid, maptrait) -> plan
    takeifftplan!(channel, level, chunkid, maptrait) -> plan

Borrow this chunk's forward or inverse FFT plan for a Toeplitz I2O step.

The map trait selects the plan. Transpose swaps the forward and inverse plans; adjoint keeps the
forward plan order and conjugates the kernel.
"""
function takefftplan! end

@doc (@doc takefftplan!) function takeifftplan! end

"""
    assertnosymmetry(collection)

Refuse a collection whose symmetry tags have not been consumed by the plain translation path.

There is no permissive default. Wrappers should forward to their contained collection; collections
that cannot carry unconsumed tags should return `nothing`.
"""
function assertnosymmetry end

# --- Traversal optimization ----------------------------------------------------------------------
#
# This hook changes traversal batching, not results.

"""
    aggregateleafchunk!(A, tree, chunknodes, moment, momentcollection, leafmul!, x, level, leafbuffer)

Apply aggregation leaf multiplication to `chunknodes`.
The default calls `aggregateleaf!` per leaf. Specialize on `leafbuffer` to batch work;
`momentcollection` is an accessor closure. Chunks can also contain internal nodes,
so implementations must filter for leaves.
"""
function aggregateleafchunk! end

# --- Near-interaction assembly -------------------------------------------------------------------
#
# `buildnearinteractionsassembler` selects an assembler; this hook executes it.

"""
    assemble(assembler::AbstractNearInteractionsAssembler, values, nearvalues, verbose=false)
    assemble(assembler::AbstractNearInteractionsAssembler, selfvalues, values, nearvalues, verbose=false)

Assemble the near block for the interaction groups supplied by the strategy.
The five-argument form separates self values; the four-argument form does not.
Concrete assemblers implement the forms they use. Missing methods raise `MethodError`;
an intentionally empty assembler must still implement its method.
"""
function assemble end

end # module Interfaces
