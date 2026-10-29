# Allocation and storage reuse

The core describes which moments are needed. Each workspace allocates its
own moments through `H2Factory.buildmoments(core)`.

## Use the pooled allocator

```julia
pools, test, trial = H2Factory.storagemomentsplan(
    testaggregationplan, trialaggregationplan, tree,
    MomentType, momentsize,
)
```

`MomentType(level)` returns a concrete type, such as `MethodMoment{1,Float64}`.
`momentsize(tree, level)` returns its array shape. Plans determine which node
moments are temporary and which must survive for later reads.

Return both the moments and their owner from the workspace builder:

```julia
return pools, H2Factory.Moments(test, trial)
```

The moment views do not keep the pools alive by themselves. Retaining only
`test` and `trial` would lose the backing storage's owner.

## Different test and trial layouts

```julia
pools, test, trial = H2Factory.storagemomentsplan(
    testaggregationplan, trialaggregationplan, tree;
    TestMomentType, TrialMomentType, testmomentsize, trialmomentsize,
)
```

Each callback describes its own side. The allocator requires compatible
scalar storage; it does not convert incompatible representations.

## Why storage can be reused

Stored moments remain available for later I2O or downward reads. Temporary
moments only survive adjacent levels. The allocator uses stored, even-level,
and odd-level pools, with alignment padding. Test and trial temporary storage
may overlap because their phases run in sequence.

This reuse depends on the plans' liveness rules. A temporary moment is not an
independent result to retain after traversal.
`H2Factory.checkmomentaliasing(A)` checks the resulting layout assumptions.

The lower-level steps are `planmomentstorage`, `allocatemomentbuffers`, and
`materializemoments`, all in H2Factory. Normal workspace construction uses
`storagemomentsplan` rather than computing pool lengths from entry counts.

See the [complete builder](../implementing/example.md#example-workspace).
