# Required methods

This checklist starts with H2Factory's workspace, linked array-backed leaf
types, and built-in transfer/coupling representations. Those choices avoid
having to implement every numerical operation yourself.

| Category | Your work |
| :--- | :--- |
| Must supply | Numerical data, core queries without defaults, and workspace builders |
| Reuse with standard fields or containers | Most accessors, leaf products, transfer/coupling application, and traversal |
| Only for extra capabilities | Adjoint leaf selection, custom representations, modulation, and specialized assembly |

The [construction diagram](../construction.md) shows when these pieces are
used.

## Must supply: method data and core queries

Construct a tree, a consistent near/far partition, leaf collections, level
transfers, and I2O couplings. Put read-only results in a subtype of
`H2Factory.GalerkinH2Core{T}` or `H2Factory.PetrovGalerkinH2Core{T}`.
H2Factory cannot derive these numerical data from an arbitrary physical operator.

For your concrete type `MyCore`, provide:

```text
H2Factory.schedulinglevels(c::MyCore)
H2Trees.isnear(c::MyCore, ::H2Factory.isForward)
H2Factory.testmomentcollections(c::MyCore, ::H2Factory.isForward)
H2Factory.trialmomentcollections(c::MyCore, ::H2Factory.isForward)
```

The predicate is already bound to the tree. Collections are indexed by leaf.
Scheduling levels must cover the core's plans, including both sides for a
block tree. A single-tree implementation can return `H2Trees.levels(c.tree)`.

## Must supply: workspace builders

| Method | Return |
| :--- | :--- |
| `H2Factory.buildmoments(c::MyCore)` | `(pools, H2Factory.Moments(test, trial))` |
| `H2Factory.buildvectorbuffers(c::MyCore)` | Leaf vector buffers; `H2Factory.VectorBuffers(test, trial)` for Petrov |
| `H2Factory.buildi2ichannels(c::MyCore)` | I2I scratch, or `H2Factory.EmptyChannels()` |
| `H2Factory.buildo2ochannels(c::MyCore)` | O2O scratch, or `H2Factory.EmptyChannels()` |
| `H2Factory.buildi2ochannel(c::MyCore)` | I2O scratch, or `H2Factory.EmptyChannels()` |

Even a method with no translation scratch needs those three channel methods.
Return fresh mutable storage for each workspace. `buildleafchannels(c)`
already defaults to `EmptyChannels()`; override it only if leaf operations
need scratch.

The [pooled allocator](../moments/allocation.md) can implement most of
`buildmoments`. [Scheduling](../maps/scheduling.md) explains chunk counts.

## Reuse: fields and plan accessors

These defaults work when your core has the corresponding fields:

| Data needed | Standard fields |
| :--- | :--- |
| Tree and complete plan bundle | `tree`, `plans` |
| Near operator | `nearinteractions` |
| I2O translation | `i2otranslator` |
| Level translators | `i2itranslator`, `o2otranslator` |
| Far prefactor, output verbosity, traversal mode | `α`, `verbose`, `farmulmode` |
| Individual plans | `testaggregationplan`, `trialaggregationplan`, `testdisaggregationplan`, `trialdisaggregationplan` |

Choose the actual values; these defaults do not construct them. Different
field layouts implement the matching [accessors](../maps/accessors.md).

In particular, keeping a bundle in `c.plans` does not automatically provide
the four individual plan accessors. Forward them explicitly:

```julia
H2Factory.testaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.testaggregationplan
H2Factory.trialaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.trialaggregationplan
H2Factory.testdisaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.testdisaggregationplan
H2Factory.trialdisaggregationplan(c::MyCore, ::H2Factory.isForward) = c.plans.trialdisaggregationplan
```

## Reuse: standard numerical operations

| Standard representation | Already supplied |
| :--- | :--- |
| Linked moment and collection types | Leaf `mul!` and conjugated products |
| Near matrix with suitable `mul!` methods | Near multiplication |
| `I2ITranslationCollection` with supported transfers | Transfer selection and application |
| Built-in I2O collection | Coupling trait, guard, and kernel |
| Ordinary moments with `NoModulation()` | No-op basis conversion |
| `H2Workspace` | Map trait, buffer accessors, scheduling, and traversal |

Names in this table belong to H2Factory except `mul!`. Reuse requires the
representation and shapes expected by that implementation. It does not
establish the accuracy of the numerical coefficients you put into it.

## Only when the capability is needed

| Capability | Additional work |
| :--- | :--- |
| Adjoint products | Declare `H2Factory.leafmul(::MyCore, ::H2Factory.isAdjoint)`; provide the appropriate prefactor and oriented numerical operations |
| Different test/trial layouts | Side-specific transfers, vector buffers, scratch, and allocation callbacks |
| Custom transfer storage | `H2Factory.leveltransfer`, plus `applytransfer!` if its result has no supported application |
| Custom I2O storage | `H2Factory.i2otrait`, `assertnosymmetry`, and `applycouplingkernel!` |
| Extra basis conversion | `H2Factory.modulate_in!`, `modulate_out!`, and matching workspace scratch |
| Custom near assembly | A strategy and assembler, unless a near operator is already supplied |

The component pages give the exact argument lists.
[Optional capabilities](optional.md) covers the related contracts; standard
Toeplitz and symmetry routes can also reuse their supplied operations.

Finally, construct `H2Factory.H2Workspace(core, scheduler)` and
[test the implementation](testing.md). The [worked example](example.md)
puts a complete core and its builders together.
