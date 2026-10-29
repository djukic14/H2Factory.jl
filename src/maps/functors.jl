# Functors used by the traversal to read node data and apply translations or leaf multiplies.

# --- Per-node moment / moment-collection getter -------------------------------------------------

"""
    NodeGetter(accessor, map)

Return `accessor(map)[node]` when called with a node.
"""
struct NodeGetter{F,A}
    accessor::F
    H2map::A
end

function (g::NodeGetter)(node)
    return g.accessor(g.H2map)[node]
end

# --- I2I translator functor ----------------------------------------------------------------------

"""
    I2ITranslatorFunctor(translator[, direction])

Apply a resolved I2I translator during aggregation. The direction defaults to
`ForwardTranslation()`. Map accessors select the right side and direction before building this
functor; adjoint aggregation uses `ConjugateTranslation()`. The plain call overwrites the output;
pass `Accumulate()` before the scheduler to add to it.
"""
struct I2ITranslatorFunctor{T,D<:AbstractTranslationDirection}
    translator::T
    direction::D
end

I2ITranslatorFunctor(translator) = I2ITranslatorFunctor(translator, ForwardTranslation())

function (f::I2ITranslatorFunctor)(
    tempmoment, storagemoment, tree, parent, child, threading::Scheduler, buffer
)
    return f(tempmoment, storagemoment, tree, parent, child, Overwrite(), threading, buffer)
end

function (f::I2ITranslatorFunctor)(
    tempmoment,
    storagemoment,
    tree,
    parent,
    child,
    accumulation::AbstractAccumulationMode,
    threading::Scheduler,
    buffer,
)
    leveltranslation!(
        tempmoment,
        storagemoment,
        f.translator,
        tree,
        parent,
        child,
        f.direction,
        accumulation,
        TranslationContext(threading, buffer),
    )
    return tempmoment
end

# --- I2O translator functor ----------------------------------------------------------------------

"""
    I2OTranslatorFunctor(map[, direction])

Apply the map's I2O translation in the given direction. The direction defaults to
`ForwardTranslation()`. The plain call overwrites the output; pass `Accumulate()` before the
scheduler to add to it.
"""
struct I2OTranslatorFunctor{A,D<:AbstractTranslationDirection}
    H2map::A
    direction::D
end

I2OTranslatorFunctor(H2map) = I2OTranslatorFunctor(H2map, ForwardTranslation())

function (f::I2OTranslatorFunctor)(
    storagemoment,
    moment,
    tree,
    receivingnode,
    translatingnode,
    threading::Scheduler,
    buffer,
)
    return f(
        storagemoment,
        moment,
        tree,
        receivingnode,
        translatingnode,
        Overwrite(),
        threading,
        buffer,
    )
end
function (f::I2OTranslatorFunctor)(
    storagemoment,
    moment,
    tree,
    receivingnode,
    translatingnode,
    accumulation::AbstractAccumulationMode,
    threading::Scheduler,
    buffer,
)
    applyi2o!(
        storagemoment,
        moment,
        H2Factory.i2otranslator(f.H2map),
        receivingnode,
        translatingnode,
        f.direction,
        accumulation,
        TranslationContext(threading, buffer),
    )
    return storagemoment
end

# --- O2O translator functor ----------------------------------------------------------------------

"""
    O2OTranslatorFunctor(translator, direction)

Apply a resolved level translator during disaggregation. The direction selects transpose or
adjoint at the transfer boundary. The plain call overwrites the output; pass `Accumulate()` before
the scheduler to add to it.
"""
struct O2OTranslatorFunctor{T,D<:AbstractTranslationDirection}
    translator::T
    direction::D
end

function (f::O2OTranslatorFunctor)(
    storagemoment, collectedmoment, tree, parent, child, threading::Scheduler, buffer
)
    return f(
        storagemoment, collectedmoment, tree, parent, child, Overwrite(), threading, buffer
    )
end

function (f::O2OTranslatorFunctor)(
    storagemoment,
    collectedmoment,
    tree,
    parent,
    child,
    accumulation::AbstractAccumulationMode,
    threading::Scheduler,
    buffer,
)
    leveltranslation!(
        storagemoment,
        collectedmoment,
        f.translator,
        tree,
        parent,
        child,
        f.direction,
        accumulation,
        TranslationContext(threading, buffer),
    )
    return storagemoment
end

# --- Leaf multiplication ------------------------------------------------------------------------
# Collections that need chunk scratch add methods to `applyleafmul!`. The defaults ignore the buffer.
function applyleafmul!(::Any, moment, momentcollection, x)
    return LinearAlgebra.mul!(moment, momentcollection, x)
end
function applyleafmul!(::Any, y, momentcollection, moment, α, β)
    return LinearAlgebra.mul!(y, momentcollection, moment, α, β)
end

"""
    LinAlgebraMulFunctor()

Leaf multiply that calls [`applyleafmul!`](@ref). The default uses `LinearAlgebra.mul!`.
"""
struct LinAlgebraMulFunctor end

function (f::LinAlgebraMulFunctor)(leafbuffer, moment, momentcollection, x)
    return applyleafmul!(leafbuffer, moment, momentcollection, x)
end

function (f::LinAlgebraMulFunctor)(leafbuffer, y, momentcollection, moment, α, β)
    return applyleafmul!(leafbuffer, y, momentcollection, moment, α, β)
end

# Adjoint maps with complex leaf collections must conjugate them as well as the transfers and coupling.

"""
    ConjugatedCollectionMul()

Leaf multiply for adjoint maps with complex leaf collections.

Aggregation computes `conj(C) * x`; disaggregation uses `adjoint(C)`. Select it with
`leafmul(A, ::isAdjoint) = ConjugatedCollectionMul()`. For real collections, select
[`LinAlgebraMulFunctor`](@ref) instead.
"""
struct ConjugatedCollectionMul end

function (::ConjugatedCollectionMul)(leafbuffer, moment, momentcollection, x)
    return applyconjugatedleafmul!(leafbuffer, moment, momentcollection, x)
end

function (::ConjugatedCollectionMul)(leafbuffer, result, momentcollection, moment, α, β)
    return applyconjugatedleafmul!(leafbuffer, result, momentcollection, moment, α, β)
end

# Direct calls without a buffer use `EmptyChannels()`.
function (f::ConjugatedCollectionMul)(moment, momentcollection, x)
    return f(EmptyChannels(), moment, momentcollection, x)
end

function (f::ConjugatedCollectionMul)(result, momentcollection, moment, α, β)
    return f(EmptyChannels(), result, momentcollection, moment, α, β)
end

# Aggregation: `conj(C) * x == conj(C * conj(x))`. This call overwrites the moment before
# conjugating it; conjugating an accumulated result would also change its previous contents.
function applyconjugatedleafmul!(::Any, moment, momentcollection, x)
    conjmul!(moment, momentcollection, x)
    for i in eachindex(moment)
        conj!(wrapvector(moment, i))
    end
    return moment
end

# Disaggregation: sum `α * adjoint(C[i]) * moment[i]` over the components. The caller has already
# conjugated `α`. Apply `β` only to the first component so the old result is scaled once.
function applyconjugatedleafmul!(::Any, result, momentcollection, moment, α, β)
    firstcomponent = first(eachindex(moment))
    LinearAlgebra.mul!(
        result,
        adjoint(momentcollection[firstcomponent]),
        wrapvector(moment, firstcomponent),
        _componentα(α, firstcomponent),
        β,
    )

    βnew = one(eltype(result))
    for i in eachindex(moment)
        i == firstcomponent && continue
        LinearAlgebra.mul!(
            result,
            adjoint(momentcollection[i]),
            wrapvector(moment, i),
            _componentα(α, i),
            βnew,
        )
    end
    return result
end

# `α` can be one scalar or one value per component.
_componentα(α, i) = α
_componentα(α::AbstractVector, i) = α[i]

"""
    ShiftedMul(inner, α, β)
    ShiftedMul(α, β)

Apply the map's leaf multiply with the caller's `α` and `β` folded into its own coefficients.
The two-argument form uses [`LinAlgebraMulFunctor`](@ref) as `inner`.
"""
struct ShiftedMul{F,α,β}
    inner::F
    α::α
    β::β
end

ShiftedMul(α, β) = ShiftedMul(LinAlgebraMulFunctor(), α, β)

function (s::ShiftedMul)(leafbuffer, y, momentcollection, incomingmoment, αA, βA)
    return s.inner(leafbuffer, y, momentcollection, incomingmoment, αA * s.α, βA + s.β)
end

# Direct calls without a buffer use `EmptyChannels()`.
function (f::LinAlgebraMulFunctor)(moment, momentcollection, x)
    return f(EmptyChannels(), moment, momentcollection, x)
end

function (f::LinAlgebraMulFunctor)(y, momentcollection, moment, α, β)
    return f(EmptyChannels(), y, momentcollection, moment, α, β)
end

function (s::ShiftedMul)(y, momentcollection, incomingmoment, αA, βA)
    return s(EmptyChannels(), y, momentcollection, incomingmoment, αA, βA)
end
