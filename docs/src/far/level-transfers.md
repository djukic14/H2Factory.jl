# I2I and O2O level transfers

I2I transfers a source moment from a child to its parent. O2O transfers a
received expansion from a parent down to a child. These connect tree levels,
unlike I2O, which connects well-separated boxes.

## Select a transfer

Your translator implements:

```text
H2Factory.leveltransfer(translator::MyTranslator, tree, parent::Int, child::Int)
```

Return the child-to-parent transfer. It can be a matrix or another supported
object. H2Factory resolves it, selects the direction, and applies it through:

```text
H2Factory.leveltranslation!(out, input, translator, tree, parent, child,
    direction, accumulation, context)
```

## Directions and accumulation

| H2Factory argument | Operation |
| :--- | :--- |
| `ForwardTranslation()` | Stored transfer |
| `TransposeTranslation()` | Transpose |
| `AdjointTranslation()` | Conjugate transpose |
| `ConjugateTranslation()` | Conjugation without transpose |
| `Overwrite()` | Replace output |
| `Accumulate()` | Add to output |

Downward traversal can use a transpose or adjoint of the stored I2I transfer.
Adjoint aggregation uses conjugation without transpose. A separate O2O data
copy is not required when the same transfers serve both directions.

## A small transfer

```jldoctest transfer-guide
using H2Factory, LinearAlgebra

struct GuideTransfer{T}
    matrix::T
end
H2Factory.leveltransfer(t::GuideTransfer, tree, parent::Int, child::Int) = t.matrix
T = ComplexF64[1+im 2; 0 3-im]
t = GuideTransfer(T)
x = ComplexF64[2, 1]
y = zeros(ComplexF64, 2)
context = H2Factory.TranslationContext(nothing, nothing)
H2Factory.leveltranslation!(y, x, t, nothing, 1, 2,
    H2Factory.ForwardTranslation(), H2Factory.Overwrite(), context)
y ≈ T * x

# output

true
```

For level/sector storage, use
`H2Factory.I2ITranslationCollection(transfers, H2Factory.SectorSelector{D}(),
H2Factory.DenseI2I())`. It reads `transfers[childlevel][sector + 1]`.
`SeparableI2I()` instead stores one factor per axis.

## The representation trait

`H2Factory.i2itrait(collection)` reads the collection's representation trait.
`DenseI2I()` selects combined transfers; `SeparableI2I()` selects separate
axis factors and their intermediate scratch. This is independent of the map
orientation.

The built-in collection declares this trait at construction. A translator
with its own `leveltransfer` method needs no new I2I trait if the returned
object already has a supported application method.

Traversal selects `Overwrite()` for a first contribution and `Accumulate()`
for later ones. These are local write modes, not the whole product's numeric
`α` and `β`. Overwrite must not read an uninitialized destination.

A new transfer representation implements
`H2Factory.applytransfer!(out, input, transfer, direction, accumulation, context)`.
Honor both modes and every direction you advertise. Local scheduling and
borrowed scratch arrive through the [context](modulation.md).
