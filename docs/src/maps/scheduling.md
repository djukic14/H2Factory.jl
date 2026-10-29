# Scheduling and multithreading

**Outer scheduling** divides tree-node visits into chunks. **Inner scheduling**
controls work inside a local kernel. Giving both all available threads is not
necessarily beneficial.

## Choose execution settings

A bare OhMyThreads scheduler gives the usual outer scheduling with serial
inner kernels:

```julia
A = H2Factory.H2Workspace(core, scheduler; nchunks=4)
```

For per-operation settings, construct
`H2Factory.defaultexecutionscheduling(levels; outer=scheduler, nchunks=4)`
or an `H2Factory.ExecutionScheduling`. Pass it in place of the scheduler.
It already carries its chunk count; a second `nchunks=` argument is rejected.

`H2Factory.Schedules` splits the core's plans into actual per-level chunks.
A small level can use fewer chunks than the requested maximum.

## Allocate one scratch buffer per chunk

Workspace builders run inside `H2Factory.withschedulechunks`. In that scope:

```julia
channels = Dict(level => H2Factory.createchannel([
    allocate_scratch(level) for _ in 1:H2Factory.nchunks(level)
]) for level in levels)
```

`allocate_scratch(level)` is your allocator and returns a fresh mutable buffer
each time. `createchannel` does not copy those buffers. It returns a chunk-indexed
`BufferStore`, not a locking `Channel`.

| Call | Meaning |
| :--- | :--- |
| `H2Factory.nchunks(level)` | Actual count in the current allocation scope |
| `H2Factory.maxnchunks(levels...)` | Largest count for a pool serving several levels |
| `H2Factory.borrowbuffer(channels, level, chunkid, trait)` | Scratch for this level and chunk |
| `H2Factory.EmptyChannels()` | No scratch needed |

`nchunks(level)` outside the scope, or for an absent level, errors. Generic
level-indexed `borrowbuffer` returns `EmptyChannels()` for a missing entry;
it does not allocate missing scratch.

## Read scratch in the kernel

Transfer and coupling kernels receive `H2Factory.TranslationContext`.
Its `scheduler` selects local execution; `buffer` holds borrowed scratch.
Level transfers borrow at the child's level; I2O uses its node's level.

For Petrov workspaces, `H2Factory.SidedChannels(test, trial)` keeps transfer
scratch separate when the sides need different sizes. Orientation routing
selects the side, not a thread ID.

## Operation tags select schedulers

Tags identify which schedule to read; they do not implement numerical work.
All names in the table belong to H2Factory.

| Lookup | Tags |
| :--- | :--- |
| Outer traversal phases | `AggregateOp`, `I2OOp`, `DisaggregateOp` |
| Inner kernels | `I2IOp`, `I2OOp`, `O2OOp`, `ModulationOp`, `ReverseModulationOp`, `LeafMulOp` |
| Specific plan node sets | `TrialAggregationOp`, `TestAggregationOp`, `TestDisaggregationOp`, `TrialDisaggregationOp` |

The outer lookup combines phase and map orientation to select the plan.
For example, forward aggregation visits the trial plan; transposed aggregation
visits the test plan.

For inner lookup, `H2Factory.scheduledop(op, trait)` swaps I2I/O2O keys under
transpose and adjoint, keeping scheduling with the corresponding tree side.
It changes the scheduler key, not the numerical kernel being applied.
Standard workspaces handle this routing. A new fast method usually needs
configuration, not new operation tags.
