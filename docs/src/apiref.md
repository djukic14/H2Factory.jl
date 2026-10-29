# API Reference

[Implementing a method](implementing/required.md) lists required methods and defaults.
For any declared hook, `H2Factory.<name>` and `H2Factory.Interfaces.<name>` refer to the
same generic function.

## Reference pages

| Page | Contents |
| --- | --- |
| [Types and data containers](reference/types.md) | Core, workspace, moment, scheduling, and translation types |
| [Functions](reference/functions.md) | Construction, accessors, traversal, and local operations |
| [Workspaces and scheduling](reference/execution.md) | Workspace construction, execution configuration, channels, and ownership guards |
| [Moment pools and aliasing](reference/pools.md) | Pooled allocation, wrapped storage, and layout checks |
| [Moment multiplication](reference/multiplication.md) | H2Factory's documented `LinearAlgebra.mul!` methods |
| [Conjugated and diagonal multiplication](reference/moment-operations.md) | `conjmul!` and `elementwisemul!` methods |
| [Type-generation macros](reference/macros.md) | Moment/collection generators and optional operator wrappers |
| [Extension declarations](reference/protocol.md) | The declared extension hooks in `H2Factory.Interfaces` |
| [Diagnostics](reference/diagnostics.md) | Structural checks, map summaries, and profiling |

## Coupling storage trait

### [`i2otrait`](@id i2otrait)

`H2Factory.i2otrait(collection)` returns the trait that selects the coupling
kernel. Built-in choices include `H2Factory.DenseTranslation()`,
`H2Factory.DiagonalTranslation()`, and `H2Factory.ToeplitzI2OTranslationTrait()`.
A wrapper must declare or forward its trait; the trait describes the stored
operator, not the map's forward/transpose/adjoint orientation.

### [`i2otranslator`](@id i2otranslator)

`H2Factory.i2otranslator(core, trait)` exposes the core's I2O translation to a
workspace. Normally it returns an `H2Factory.I2OTranslation` combining a coupling
collection and modulation policy. The one-argument map accessor resolves its
orientation and forwards to the core through the workspace.

## Level-transfer application

### [`applytransfer!`](@id applytransfer!)

`H2Factory.applytransfer!(out, in, transfer, direction, accumulation, context)`
applies an already resolved level transfer. The generic implementation selects
the requested direction, then calls `H2Factory.applydirectedtransfer!`.
Dense matrices and `SeparableTransfer` have built-in implementations. A custom
representation must honor both overwrite and accumulation semantics. This is
separate from I2O coupling application.

## Pool sizing internals

### [`calculatememoryrequirements`](@id calculatememoryrequirements)

`H2Factory.calculatememoryrequirements` is the internal sizing step used by
`allocatemomentbuffers`. It accounts for moment sizes and alignment while
walking the aggregation plans. Its returned groups depend on the plan type.
For method construction, use `H2Factory.storagemomentsplan` rather than calling
this internal step or deriving pool lengths by adding node-entry counts.

### [Leaf-multiply scheduling key](@id H2Factory.LeafMulOp)

`H2Factory.LeafMulOp()` identifies leaf multiplication in execution scheduling.
It is an internal operation tag, not a leaf-multiply implementation. A core can
specialize `H2Factory.leafmulscheduler` to choose the leaf kernel's inner
scheduler when constructing a workspace from a bare scheduler.

## Alphabetical index

```@index
```
