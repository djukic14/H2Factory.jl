# Moments and moment collections

A **moment collection** holds matrices built by the fast method. At a leaf,
they connect the original basis coefficients to expansion coefficients.
A **moment** holds those expansion coefficients for one multiplication.

```text
trial coefficients x -> trial collection -> source moment
                                                |
                                         far interactions
                                                |
test coefficients y <- test collection <- received moment
```

Collections live in the core. Moments live in the workspace and change on
each product. Both can have several components, such as vector and divergence
components of an expansion.

## The leaf operations

For your own representation, the hooks are:

```text
H2Factory.applyleafmul!(scratch, moment, collection, x)
H2Factory.applyleafmul!(scratch, y, collection, moment, α, β)
```

The first converts leaf input to a moment. The second writes output with
scaling and accumulation. Their defaults call `LinearAlgebra.mul!`.
`H2Factory.applyconjugatedleafmul!` has the same two call shapes and delegates
to `H2Factory.conjmul!` by default.

The [type-generation macros](types.md) provide these products for linked
array-backed types. For vector output, their ordinary product uses the
**transpose** of each collection matrix, not its adjoint. The map's
[orientation handling](../maps/accessors.md) chooses conjugation separately.

## Computing collections

Your construction code computes each leaf's matrices and exposes them through
`H2Factory.testmomentcollections(core, H2Factory.isForward())` and
`H2Factory.trialmomentcollections(core, H2Factory.isForward())`.
The returned containers are indexed by leaf node.

For integration-based methods, `H2Factory.assemblemoments!` is available through
the [BEAST extension](../extensions.md). Its storer writes assembled values
into collections. A method using analytic coefficients need not use it.

Next: [create and link types](types.md), then [allocate moments](allocation.md).
