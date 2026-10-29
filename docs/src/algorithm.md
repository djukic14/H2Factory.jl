# How the multiplication works

A product combines a directly evaluated near part and a hierarchical far part:

```text
y = near interactions * x + far interactions * x
```

The **trial** side is the input side, where x lives. The **test** side is the
output side, where y is computed. In Galerkin maps they share a space and tree;
Petrov maps may use different ones.

See [Construction and execution](construction.md) for the lifecycle of the
core and workspace.

## Follow the far product

![Source moments are formed at leaves and transferred to their parents.](assets/aggregationtree.svg)

At each source leaf, a [moment collection](moments/overview.md) turns input
coefficients into expansion coefficients: a moment. [I2I transfers](far/level-transfers.md)
combine child moments into parent moments. A small parent expansion can
then represent many leaves.

![I2O contributions enter the receiving tree and move down towards its leaves.](assets/disaggregationtranslationtree.svg)

[I2O coupling](far/coupling.md) connects well-separated source and receiving
nodes. [O2O transfers](far/level-transfers.md) pass received contributions
towards the leaves. The test-side collections turn those expansions into y.

Some methods need [basis conversion](far/modulation.md) around I2O.
Toeplitz coupling, for example, operates in the frequency domain.

## Who does what?

| Part | H2Factory executes | Your method supplies |
| :--- | :--- | :--- |
| Near interactions | `nearmul!` | A near operator supporting `mul!` |
| Leaves | `applyleafmul!` | Collections and compatible moment operations |
| Between levels | `leveltranslation!` | Transfer selection and data |
| Between boxes | `applyi2o!` | Couplings and their application |
| Basis conversion | `modulate_in!`, `modulate_out!` | A suitable policy, or no conversion |

The functions in the middle column belong to H2Factory. Numerical data is
read from the [core](maps/core-workspace.md); changing moments and scratch
belong to the workspace.

## Plans, then execution

H2Trees plans decide which nodes and pairs each phase visits. H2Factory
schedules those visits and applies the numerical operations.

There are two H2Trees modes: `AggregateMode()` puts I2O in the downward pass,
as above; `AggregateTranslateMode()` puts it in the upward pass.
[Aggregation and disaggregation](far/traversal.md) explains their execution.
The [H2Trees plans guide](https://djukic14.github.io/H2Trees.jl/dev/plans/plans/)
covers construction.

For a complete implementation, follow [Required methods](implementing/required.md).
