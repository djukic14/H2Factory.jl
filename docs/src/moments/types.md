# Creating and linking types

The macros create array-backed containers and their products. They do not
assemble the fast method's leaf matrices.

## Create a linked pair

```jldoctest linked-guide
using H2Factory, LinearAlgebra

@createlinked GuideMoment GuideCollection
m = GuideMoment{1,Float64}((2,))
C = GuideCollection{1,Float64}((2, 3))
C[1] .= [1 0 2; 0 1 3]
x = [1.0, 2.0, 3.0]
mul!(m, C, x)
m[1] == C[1] * x

# output

true
```

`1` is the component count, not the spatial dimension. The moment has two
coefficients; the collection has one matrix with two rows and three columns.
Shape constructors allocate uninitialized arrays. Fill collection matrices
before applying them, and initialize moments before accumulation.

The reverse product `mul!(y, C, m)` sums
`transpose(C[i]) * vec(m[i])` over components. The five-argument form also
accepts `α` and `β`.

## Link separately created types

```julia
@createmoment MethodMoment
@createmomentcollection TestCollection
@createmomentcollection TrialCollection
@linkmomenttocollection MethodMoment TestCollection
@linkmomenttocollection MethodMoment TrialCollection
```

These macros belong to H2Factory. Generated constructors can also wrap
existing `StaticArrays.SVector` arrays without copying them. That shares
mutable storage; it does not create an independent workspace.

## Conjugated products

Linked types provide `H2Factory.conjmul!`. For moment output, it conjugates
the input vector; for vector output, it conjugates the transposed product.
The scaled forms are in the [multiplication reference](../reference/moment-operations.md).

Custom representations can implement the leaf hooks directly. Supporting
forward multiplication alone does not establish transpose or adjoint support.
