# The near-interaction strategies and assemblers of earlier releases, kept for MLFMA and
# MultilevelKernelInterpolation. Each assembler maps onto an `AssemblyStrategy`; calling
# `assembler(tree, isnear; verbose=false)` assembles through `assemblenearinteractions`. The
# type parameter `T` of an assembler and `γ` are kept for the constructor shapes only:
# entries have the scalar type of the kernel matrix.

# --- Assemblers ---------------------------------------------------------------------------

"""
    AbstractNearInteractionsAssembler{T}

Base type for near-interaction assemblers. Calling `assembler(tree, isnear; verbose=false)`
assembles through `assemblenearinteractions` with the kernel matrix of
`operator(assembler)`, `testspace(assembler)` and `trialspace(assembler)` and the
`assemblystrategy(assembler)`. The accessors read the fields `operator`, `testspace`,
`trialspace` and `quadstrat`; an assembler with other fields overrides them.
"""
abstract type AbstractNearInteractionsAssembler{T} end

"""
    SparseBEASTNearInteractionsAssembler{T}(operator, testspace, trialspace, quadstrategy)

Assemble near interactions into a `SparseMatrixCSC` with static scheduling, through
`AssemblyStrategy(quadstrategy, SparseStorage())`.
"""
struct SparseBEASTNearInteractionsAssembler{T,Operator,TestSpace,TrialSpace,QuadStrat} <:
       AbstractNearInteractionsAssembler{T}
    operator::Operator
    testspace::TestSpace
    trialspace::TrialSpace
    quadstrat::QuadStrat

    function SparseBEASTNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end

"""
    BlockBEASTNearInteractionsAssembler{T}(operator, testspace, trialspace, quadstrategy)

Assemble near interactions into a `BlockSparseMatrix` with dynamic scheduling, through
`AssemblyStrategy(quadstrategy, BlockSparseStorage())`.
"""
struct BlockBEASTNearInteractionsAssembler{T,Operator,TestSpace,TrialSpace,QuadStrat} <:
       AbstractNearInteractionsAssembler{T}
    operator::Operator
    testspace::TestSpace
    trialspace::TrialSpace
    quadstrat::QuadStrat

    function BlockBEASTNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end

"""
    DummyNearInteractionsAssembler{T}(operator, testspace, trialspace, quadstrategy)

Assemble an empty `SparseMatrixCSC` without walking the tree:
`AssemblyStrategy(DummyEntries(), SparseStorage())`. `quadstrategy` is not used.
"""
struct DummyNearInteractionsAssembler{T,Operator,TestSpace,TrialSpace,QuadStrat} <:
       AbstractNearInteractionsAssembler{T}
    operator::Operator
    testspace::TestSpace
    trialspace::TrialSpace
    quadstrat::QuadStrat

    function DummyNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end

operator(assembler::AbstractNearInteractionsAssembler) = assembler.operator
testspace(assembler::AbstractNearInteractionsAssembler) = assembler.testspace
trialspace(assembler::AbstractNearInteractionsAssembler) = assembler.trialspace
quadstrategy(assembler::AbstractNearInteractionsAssembler) = assembler.quadstrat

# The schedulers reproduce `Threads.@threads :static` and `Threads.@threads` of the
# assemblers these replace.
function assemblystrategy(assembler::SparseBEASTNearInteractionsAssembler)
    return AssemblyStrategy(quadstrategy(assembler), SparseStorage(), StaticScheduler())
end

function assemblystrategy(assembler::BlockBEASTNearInteractionsAssembler)
    return AssemblyStrategy(
        quadstrategy(assembler), BlockSparseStorage(), DynamicScheduler()
    )
end

function assemblystrategy(::DummyNearInteractionsAssembler)
    return AssemblyStrategy(DummyEntries(), SparseStorage(), SerialScheduler())
end

# Interaction strategy around a near predicate that is already evaluated on the tree.
struct NearPredicate{F} <: AbstractInteractionStrategy
    isnear::F
end

nearcriterion(strategy::NearPredicate) = Returns(strategy.isnear)

function (assembler::AbstractNearInteractionsAssembler)(tree, isnear; verbose::Bool=false)
    return assemblenearinteractions(
        operator(assembler),
        testspace(assembler),
        trialspace(assembler),
        tree,
        assemblystrategy(assembler),
        NearPredicate(isnear);
        verbose=verbose,
    )
end

isgalerkinsymmetric(operator) = isgalerkinsymmetric(typeof(operator))
isgalerkinsymmetric(::Type) = false

# --- Strategies ---------------------------------------------------------------------------

"""
    AbstractNearInteractionsStrategy

Base type for strategies that select a near-interaction assembler through
[`buildnearinteractionsassembler`](@ref).
"""
abstract type AbstractNearInteractionsStrategy end

"""
    BlockBEASTNearInteractionsStrategy(quadstrategy)

Select [`BlockBEASTNearInteractionsAssembler`](@ref).
"""
struct BlockBEASTNearInteractionsStrategy{Q} <: AbstractNearInteractionsStrategy
    quadstrategy::Q
end

"""
    DummyNearInteractionsStrategy(quadstrategy)

Select [`DummyNearInteractionsAssembler`](@ref), which returns an empty near block.
"""
struct DummyNearInteractionsStrategy{Q} <: AbstractNearInteractionsStrategy
    quadstrategy::Q
end

function buildnearinteractionsassembler(
    operator, testspace, trialspace, γ, strategy::BlockBEASTNearInteractionsStrategy
)
    return BlockBEASTNearInteractionsAssembler{typeof(γ)}(
        operator, testspace, trialspace, strategy.quadstrategy
    )
end

function buildnearinteractionsassembler(
    operator, testspace, trialspace, γ, strategy::DummyNearInteractionsStrategy
)
    return DummyNearInteractionsAssembler{typeof(γ)}(
        operator, testspace, trialspace, strategy.quadstrategy
    )
end

function assemblenearinteractions(
    operator,
    testspace,
    trialspace,
    tree,
    γ,
    nearinteractionsstrategy::AbstractNearInteractionsStrategy,
    interactionstrategy::AbstractInteractionStrategy;
    verbose=false,
)
    assembler = buildnearinteractionsassembler(
        operator, testspace, trialspace, γ, nearinteractionsstrategy
    )
    return assembler(tree, nearcriterion(interactionstrategy)(tree); verbose=verbose)
end
