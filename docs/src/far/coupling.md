# I2O coupling

I2O takes a source moment and contributes to a well-separated receiving node.
The coupling collection stores or computes that operation for each pair.
Building those coefficients is the fast method's responsibility.

## Use a built-in collection

For ordinary dense couplings:

```julia
c = H2Factory.I2OTranslationCollection(blocks, H2Factory.DenseTranslation())
translation = H2Factory.I2OTranslation(c)
```

`blocks` maps `(receivingnode, translatingnode)` in forward order to coupling
matrices. Unlike near blocks, these matrices act on expansion coefficients,
not directly on the original basis-function indices.

The tree-based constructor can compute each unique translation once:

```text
H2Factory.I2OTranslationCollection(tree, translatingplan, reuse, builder;
    i2otranslationtrait=H2Factory.DenseTranslation(), verbose=false)
```

It returns `(collection, EmptyChannels())`. The builder provides
`eltype(builder)` and `builder(receiver, source, geometricdirection)`.
`reuse` is an H2Trees translation-sharing trait. It is distinct from
`i2otranslationtrait`, which selects the application kernel.

## Apply one pair

`H2Factory.i2otrait(c)` returns a coupling-storage trait, such as
`DenseTranslation()`, `DiagonalTranslation()`, or `ToeplitzI2OTranslationTrait()`.
`applycoupling!` resolves it before calling `applycouplingkernel!`.

In contrast, H2Trees' `DirectionInvariance()` and
`DirectionInvariancePerLevel()` select which geometric displacements share
data. They do not choose dense versus Toeplitz application.
`SymmetryDirectionInvariancePerLevel` also retains tags requiring
[symmetry handling](symmetry.md).

Direction and accumulation are separate kernel arguments. The same dense
collection can run forward, transposed, or adjointed, with either write mode.

```text
H2Factory.applyi2o!(out, input, translation, receiver, source,
    direction, accumulation, context)
```

This call applies the coupling only. [Modulation](modulation.md) is performed
separately, around the node's group of pair interactions.

## Implement a new collection

Derive from `H2Factory.AbstractI2OCouplingCollection` and supply:

```text
H2Factory.assertnosymmetry(c::MyCouplings)
H2Factory.i2otrait(c::MyCouplings)
H2Factory.Interfaces.applycouplingkernel!(out, input, c::MyCouplings, trait,
    receiver::Int, source::Int, direction, accumulation, context)
```

`assertnosymmetry` must reject unhandled symmetry tags; it can return `nothing`
for a representation with none. `i2otrait` selects your kernel specialization.
The kernel changes `out`, returns it, and respects overwrite/accumulate.

`H2Factory.couplingkey(receiver, source, direction)` selects the stored pair;
`H2Factory.directedoperation(direction)` selects identity, transpose, or adjoint.
Handle both if writing a matrix-backed kernel. Swapping a pair alone does not
transpose its coupling.

Extend `applycouplingkernel!`, not `applycoupling!` or `applyi2o!`: those supplied
callers keep the symmetry guard in the execution path.

## A custom matrix-backed kernel

This small example checks all three directions and both accumulation modes.
It needs no tree construction or physical operator assembly.

```jldoctest coupling-guide
using H2Factory, LinearAlgebra

struct GuideCouplingTrait <: H2Factory.AbstractI2OTranslationTrait end
struct GuideCouplings{B} <: H2Factory.AbstractI2OCouplingCollection
    blocks::B
end
H2Factory.assertnosymmetry(::GuideCouplings) = nothing
H2Factory.i2otrait(::GuideCouplings) = GuideCouplingTrait()

function H2Factory.applycouplingkernel!(out, input, c::GuideCouplings,
    ::GuideCouplingTrait, receiver::Int, source::Int, direction,
    mode::Union{H2Factory.Overwrite,H2Factory.Accumulate},
    context::H2Factory.TranslationContext)
    key = H2Factory.couplingkey(receiver, source, direction)
    block = H2Factory.directedoperation(direction)(c.blocks[key])
    return mul!(out, block, input, true, mode isa H2Factory.Accumulate)
end

K = ComplexF64[1+im 2 3-im; -1 4im 2]
translation = H2Factory.I2OTranslation(GuideCouplings(Dict((2, 1) => K)))
context = H2Factory.TranslationContext(nothing, nothing)
directions = (
    (H2Factory.ForwardTranslation(), identity, 2, 1),
    (H2Factory.TransposeTranslation(), transpose, 1, 2),
    (H2Factory.AdjointTranslation(), adjoint, 1, 2),
)
all(directions) do (direction, operation, receiver, source)
    block = operation(K)
    input = fill(1.0 + im, size(block, 2))
    out = fill(7.0 - 2im, size(block, 1))
    H2Factory.applyi2o!(out, input, translation, receiver, source,
        direction, H2Factory.Overwrite(), context)
    overwritten = out ≈ block * input
    H2Factory.applyi2o!(out, input, translation, receiver, source,
        direction, H2Factory.Accumulate(), context)
    overwritten && out ≈ 2 * (block * input)
end

# output

true
```
