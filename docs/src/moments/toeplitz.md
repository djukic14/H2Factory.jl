# Toeplitz moments

A Toeplitz product needs space for its circulant embedding. A generated
Toeplitz moment holds ordinary component arrays and padded component arrays.

## Create the types

```julia
@createtoeplitzmoment MethodMoment
@createmomentcollection MethodCollection
@linkmomenttocollection MethodMoment MethodCollection
```

`@createlinkedtoeplitz MethodMoment MethodCollection` combines these steps.
The macros belong to H2Factory; the default moment parent is
`H2Factory.AbstractToeplitzMoment`.

`MethodMoment{D,T}(shape)` allocates `D` ordinary arrays and `D` padded arrays.
Padding defaults to `2 .* shape .- 1`; a two-shape constructor accepts an
explicit padded shape. The arrays are initially uninitialized.

| Access | Result |
| :--- | :--- |
| `moment[i]` | Ordinary component array |
| `H2Factory.toeplitzmoment(moment, i)` | Padded component array |
| `H2Factory.toeplitzvector(moment, i)` | Cached flat view of the padded array |

Pooled allocation supports these layouts too. Keep its backing pools alive
just as for ordinary moments.

## Storage is not the operation

The moment type provides storage. The I2O modulation policy selects conversion.
The standard `H2Factory.ToeplitzModulation()` performs FFT conversion; a custom
two-basis method may use the same two-array layout with a different policy.

`NoModulation()` with an `AbstractToeplitzMoment` is rejected. H2Factory does
not assume that conversion has already happened merely because padded arrays
exist. See [Toeplitz coupling](../far/toeplitz.md) and [modulation](../far/modulation.md).
