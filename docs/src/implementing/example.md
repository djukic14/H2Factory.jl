# Worked example

This example builds a complete Galerkin map: tree, leaf collections,
translations, core, and workspace. It uses H2Factory, H2Trees, StaticArrays,
and OhMyThreads, without helpers from H2FactoryTestUtils.

The code blocks run in order in one Julia session. They are also doctests,
checked whenever these docs are built.

## [1. Build the tree and numerical data](@id example-data)

We use 32 points on a line and the kernel

```math
K(s,t) = 1 + 2s + 3t + (4+\mathrm{i})st.
```

For a group of source points, only two coefficients are needed:
`sum(x[j])` and `sum(coordinates[j] * x[j])`. Each leaf collection computes
these two sums. A parent adds its children's sums, so the level transfer is
the identity matrix.

The I2O matrix `C` turns these source coefficients into the coefficients
`a, b` of the received field `a + b*s`. The test leaf collection then evaluates
that field at the receiving points. This kernel is chosen to make the
two-coefficient representation exact; no approximation tolerance needs tuning.

```jldoctest workedexample
using H2Factory
import H2Trees
using LinearAlgebra
using OhMyThreads: SerialScheduler
using StaticArrays: SVector

@createlinked WorkedMoment WorkedCollection

coordinates = collect(range(0.0, 1.0; length=32))
points = [SVector(t) for t in coordinates]
tree = H2Trees.buildtree(points; builder=H2Trees.TwoNTreeBuilder(; minvalues=2))
n = length(coordinates)
kernel(s, t) = 1 + 2s + 3t + (4 + im) * s * t

strategy = H2Factory.InteractionStrategy()
plans = H2Factory.assembleplans(tree, strategy)
nearpredicate = H2Factory.nearcriterion(strategy)(tree)
n

# output

32
```

Each leaf basis has a row of ones and a row of point coordinates. The
transfer returns the same identity matrix for every tree edge:

```jldoctest workedexample
function leafcollection(tree, leaf, coordinates)
    indices = H2Trees.values(tree, leaf)
    collection = WorkedCollection{1,ComplexF64}((2, length(indices)))
    collection[1][1, :] .= 1
    collection[1][2, :] .= coordinates[indices]
    return collection
end
collections = Dict(leaf => leafcollection(tree, leaf, coordinates) for leaf in H2Trees.leaves(tree))

struct WorkedTransfer{T}
    matrix::T
end
H2Factory.leveltransfer(t::WorkedTransfer, tree, parent::Int, child::Int) = t.matrix
transfer = WorkedTransfer(Matrix{ComplexF64}(I, 2, 2))
size(transfer.matrix)

# output

(2, 2)
```

Every far pair selected by the forward plan uses `C`. Near interactions are
assembled directly from the kernel:

```jldoctest workedexample
C = ComplexF64[1 3; 2 4+im]
plan = plans.testdisaggregationplan
blocks = Dict(
    (receiver, source) => C for level in H2Trees.levels(plan) for
    receiver in H2Trees.receivingnodes(plan, level) for
    source in H2Trees.translatingnodes(plan, receiver, level)
)
couplings = H2Factory.I2OTranslationCollection(blocks, H2Factory.DenseTranslation())
translation = H2Factory.I2OTranslation(couplings)

near = zeros(ComplexF64, n, n)
rowgroups, colgroups = H2Trees.nearinteractions(tree; isnear=nearpredicate)
for (rows, cols) in zip(rowgroups, colgroups)
    near[rows, cols] .= [kernel(coordinates[i], coordinates[j]) for i in rows, j in cols]
end
@assert !isempty(blocks)
@assert any(!H2Trees.isleaf(tree, source) for (_, source) in keys(blocks))
@assert 0 < count(x -> !iszero(x), near) < n^2
size(near)

# output

(32, 32)
```

The basis matrix at a leaf has shape `(2, number_of_leaf_points)`. Every
transfer and coupling has shape `(2, 2)`. `blocks` contains only the I2O
pairs visited by the forward plan. All pairs use the same read-only `C` here.

The three assertions check that the tree has far interactions, that some
source moments require aggregation above the leaves, and that the near matrix
is neither empty nor the full matrix. The near matrix is stored densely to
keep this small example simple.

## [2. Define the core](@id example-core)

The core stores the data built above. Fields with standard names use
H2Factory's default accessors. The remaining accessors are defined explicitly.

```jldoctest workedexample
struct MyCore{Tree,Near,Plans,Collections,Transfer,Coupling} <: H2Factory.GalerkinH2Core{ComplexF64}
    tree::Tree
    isnear::Near
    plans::Plans
    momentcollections::Collections
    i2itranslator::Transfer
    o2otranslator::Transfer
    i2otranslator::Coupling
    nearinteractions::Matrix{ComplexF64}
    α::ComplexF64
    verbose::Bool
    farmulmode::H2Trees.AggregateMode
end

H2Factory.schedulinglevels(c::MyCore) = H2Trees.levels(c.tree)
H2Trees.isnear(c::MyCore, ::H2Factory.isForward) = c.isnear
H2Factory.testmomentcollections(c::MyCore, ::H2Factory.isForward) = c.momentcollections
H2Factory.trialmomentcollections(c::MyCore, ::H2Factory.isForward) = c.momentcollections
H2Factory.testaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.testaggregationplan
H2Factory.trialaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.trialaggregationplan
H2Factory.testdisaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.testdisaggregationplan
H2Factory.trialdisaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.trialdisaggregationplan
H2Factory.leafmul(::MyCore, ::H2Factory.isAdjoint) = H2Factory.ConjugatedCollectionMul()
H2Factory.α(c::MyCore, ::H2Factory.isAdjoint) = conj(c.α)

core = MyCore(tree, nearpredicate, plans, collections, transfer, transfer,
    translation, near, 1.0 + 0im, false, H2Trees.AggregateMode())
eltype(core) === ComplexF64

# output

true
```

Both sides use the same leaf collections. The identity transfer also serves
both I2I and O2O. `α` is the far-field prefactor, set to one; `verbose=false`
disables progress output. The two adjoint methods choose conjugating leaf
multiplication and conjugate the prefactor.

The [implementation checklist](required.md) explains these
accessors individually.

## [3. Allocate the workspace](@id example-workspace)

Each node moment has one component array with two coefficients. The pooled
allocator returns both the moments and the memory that keeps them alive.

The vector buffers have one entry per leaf point. The dense translations need
no additional scratch, so all three channel builders return `EmptyChannels()`.

```jldoctest workedexample
function H2Factory.buildmoments(c::MyCore)
    pools, test, trial = H2Factory.storagemomentsplan(
        c.plans.testaggregationplan, c.plans.trialaggregationplan, c.tree,
        level -> WorkedMoment{1,ComplexF64}, (tree, level) -> (2,))
    return pools, H2Factory.Moments(test, trial)
end
H2Factory.buildvectorbuffers(c::MyCore) = Dict(
    leaf => zeros(ComplexF64, length(H2Trees.values(c.tree, leaf))) for
    leaf in H2Trees.leaves(c.tree)
)
H2Factory.buildi2ichannels(::MyCore) = H2Factory.EmptyChannels()
H2Factory.buildo2ochannels(::MyCore) = H2Factory.EmptyChannels()
H2Factory.buildi2ochannel(::MyCore) = H2Factory.EmptyChannels()

A = H2Factory.H2Workspace(core, SerialScheduler(); nchunks=1)
size(A)

# output

(32, 32)
```

`A` is now the executable map. The [workspace guide](../maps/core-workspace.md)
describes what it shares with the core; [scratch and scheduling](../maps/scheduling.md)
covers methods that need temporary buffers or more chunks.

## [4. Multiply and check the result](@id example-check)

The dense reference is assembled directly from `kernel`, without using the
leaf collections, transfers, or I2O matrix. The input has nonconstant real
and imaginary parts.

```jldoctest workedexample
F = [kernel(s, t) for s in coordinates, t in coordinates]
x = ComplexF64[sin(j) + im * cos(2j) for j in 1:n]
y = zeros(ComplexF64, n)
mul!(y, A, x)

checks = (
    isapprox(y, F * x; rtol=1e-12),
    isapprox(transpose(A) * x, transpose(F) * x; rtol=1e-12),
    isapprox(adjoint(A) * x, adjoint(F) * x; rtol=1e-12),
    isapprox(Matrix(A), F; rtol=1e-12),
)
checks

# output

(true, true, true, true)
```

The first three checks cover forward, transpose, and adjoint products. The
last reconstructs every column of `A` and compares the full matrix.

A second workspace can use the same core with separate buffers:

```jldoctest workedexample
B = H2Factory.similarworkspace(A)
(
    H2Factory.core(B) === H2Factory.core(A),
    H2Factory.workspacebuffers(B) !== H2Factory.workspacebuffers(A),
    isapprox(B * x, F * x; rtol=1e-12),
)

# output

(true, true, true)
```

```@raw html
<details><summary>Doctest: scaled multiplication in all three orientations</summary>
```

```jldoctest workedexample
all(((A, F), (transpose(A), transpose(F)), (adjoint(A), adjoint(F)))) do (map, reference)
    out = fill(2.0 + im, n)
    expected = (2 - im) * (reference * x) + (3 + 2im) * out
    mul!(out, map, x, 2 - im, 3 + 2im)
    isapprox(out, expected; rtol=1e-12)
end

# output

true
```

```@raw html
</details>
```
