# Moment multiplication

[API reference](../apiref.md) | [Moments and leaf collections](../moments/overview.md)

These are the `LinearAlgebra.mul!` methods documented by H2Factory, not the full
LinearAlgebra reference. The linked collection-to-output operation uses a
transpose; conjugated leaf multiplication is selected separately for adjoint
maps. See [Transpose and adjoint](../maps/accessors.md).

```@autodocs
Modules = [H2Factory]
Order = [:function]
Filter = object -> object === H2Factory.LinearAlgebra.mul!
```
