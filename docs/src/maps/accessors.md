# Accessors and orientation

Accessors let a core choose its own field layout. Traversal asks for the
tree, collections, and plans through these functions rather than inspecting
your struct directly.

One-argument map calls resolve `transpose(A)` and `adjoint(A)` first.
Implement the core methods below rather than replacing that map-level routing.

## The map trait

`H2Factory.maptrait(A)` delegates to `maptrait(typeof(A))` and returns
`isForward()`, `isTranspose()`, or `isAdjoint()`. These describe the whole
map, not an individual transfer.

Subtypes of `H2Factory.AbstractH2Map` already default to forward orientation;
standard transpose and adjoint wrappers are handled automatically. A core
is data, not an executable map, so it does not need its own map trait.
Custom map wrappers outside this routing need an explicit type-level method
and a way to reach their forward map.

## Defaults you can reuse

| Core call | Default |
| :--- | :--- |
| `H2Trees.tree(c, isForward())` | `c.tree` |
| `H2Factory.plans(c)` | `c.plans` |
| `H2Factory.nearinteractions(c, isForward())` | `c.nearinteractions` |
| `H2Factory.i2otranslator(c, trait)` | `c.i2otranslator` |
| `H2Factory.triali2itranslator(c)` / `testi2itranslator(c)` | `c.i2itranslator` |
| `H2Factory.trialo2otranslator(c)` / `testo2otranslator(c)` | `c.o2otranslator` |
| `H2Factory.farmulmode(c, isForward())` | `c.farmulmode` |
| `H2Factory.α(c, trait)` | `c.α`, unchanged |
| `H2Factory.verbose(c, trait)` | `c.verbose` |

The traits belong to H2Factory: `isForward()`, `isTranspose()`, and
`isAdjoint()`. Petrov cores can return separate test and trial transfers.

## Plans and collections

Expose `testmomentcollections`, `trialmomentcollections`,
`testaggregationplan`, `trialaggregationplan`, `testdisaggregationplan`, and
`trialdisaggregationplan` as `(c::MyCore, ::H2Factory.isForward)` methods
on H2Factory's generics. Default plan methods read fields with those names
directly on the core. If plans live in `c.plans`, forward to them explicitly.

H2Factory swaps the sides for transpose and adjoint. `H2Trees.isnear(c,
::H2Factory.isForward)` returns the predicate already bound to the core's tree.
The [checklist](../implementing/required.md) shows the complete calls.

## Adjoint needs an explicit leaf operation

The map trait and traversal phase together select the local direction:

| Map | Upward I2I | I2O | Downward O2O |
| :--- | :--- | :--- | :--- |
| Forward | Stored transfer | Stored coupling | Transpose transfer |
| Transpose | Stored transfer on swapped side | Transpose coupling | Transpose transfer on swapped side |
| Adjoint | Conjugated transfer on swapped side | Adjoint coupling | Adjoint transfer on swapped side |

Adjoint aggregation uses `ConjugateTranslation()`, not
`AdjointTranslation()`: it still moves child coefficients to their parent.
The [transfer guide](../far/level-transfers.md#Directions-and-accumulation)
lists these direction arguments.

Forward and transpose default to `H2Factory.LinAlgebraMulFunctor()`.
For linked complex collections:

```julia
H2Factory.leafmul(::MyCore, ::H2Factory.isAdjoint) =
    H2Factory.ConjugatedCollectionMul()
```

Real collections can use `LinAlgebraMulFunctor()` for adjoint too. A complex
map element type does not imply complex collections; choose from the actual
leaf representation. Custom representations supply the corresponding leaf hooks.

The default prefactor is not conjugated automatically. For a complex scalar:

```julia
H2Factory.α(c::MyCore, ::H2Factory.isAdjoint) = conj(c.α)
```

See [level transfers](../far/level-transfers.md) for translation directions.
