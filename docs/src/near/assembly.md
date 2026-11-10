# Computing near interactions

Near interactions are evaluated directly rather than through the far-field
expansion. There are three choices: which interactions are near, how their
entries are computed, and how the results are stored.

## Select the interactions

```julia
interaction = H2Factory.InteractionStrategy(isnear=nearfunctor)
plans = H2Factory.assembleplans(tree, interaction)
predicate = H2Factory.nearcriterion(interaction)(tree)
```

`nearfunctor` is a tree-independent H2Trees criterion. Calling it on `tree`
binds it to that geometry. Use the same partition for near assembly and far
plans; otherwise interactions can be omitted or counted twice.

`InteractionStrategy` also sets the translating-node iterator, aggregation-node
selection, and traversal mode. It belongs to the whole interaction partition,
not only the near field. H2Trees documents the geometric predicates and plans.

## Assemble the near interactions

```text
H2Factory.assemblenearinteractions(op, testspace, trialspace, tree,
    assemblystrategy, interaction; verbose=false)
```

The operator and spaces define a kernel matrix and its scalar type. The
assembly strategy says how its entries are evaluated, how they are stored and
how the work is scheduled:

```text
H2Factory.AssemblyStrategy(matrixdata, storage; scheduler=DynamicScheduler())
```

- `matrixdata` is passed to `H2Factory.buildkernelmatrix(op, testspace,
  trialspace, matrixdata)`. For a BEAST operator it is the quadrature strategy,
  and the [H2BEAST extension](../extensions.md) returns a
  `H2Factory.BEASTKernelMatrix`.
- `storage` is `H2Factory.SparseStorage()` for a `SparseMatrixCSC` or
  `H2Factory.BlockSparseStorage()` for a `BlockSparseMatrix`, which needs the
  H2BlockSparse extension.
- `scheduler` is the OhMyThreads scheduler that distributes the near blocks.

A kernel matrix built elsewhere, such as a `H2Factory.PointKernelMatrix`, is
assembled with the same strategies:

```text
H2Factory.assemblenearinteractions(kernelmatrix, tree, assemblystrategy,
    interaction; verbose=false)
```

## Write a kernel matrix

A kernel matrix subtypes `H2Factory.AbstractKernelMatrix{T}`, implements
`size`, and adds its entries to a block:

```text
(kernelmatrix::MyKernelMatrix)(block, rows, columns)
```

`rows` and `columns` are global indices, and `block[m, n]` receives the entry
for `rows[m]` and `columns[n]`. Blocks are evaluated concurrently, so the call
must be safe for concurrent tasks. `H2Factory.buildkernelmatrix` methods for
the kernel family map an operator, spaces and matrix data onto the matrix.

## Use existing data

`H2Factory.DummyEntries()` as matrix data selects a
`H2Factory.DummyKernelMatrix`. It has no near blocks, so assembly walks no
tree and returns an empty matrix in the selected storage. Use it for tests or
an intentionally far-only map; it is not a complete physical near field.

`H2Factory.RandomEntries()` selects a `H2Factory.RandomKernelMatrix` of the
same size and scalar type. Assembly walks the tree as for the real kernel, so
the near matrix has the real sparsity, but its entries are random. Use it to
check or measure storage and multiplication without evaluating the kernel.

An already assembled near matrix can be stored on the core directly; it must
match the map's tree and spaces.

## Strategies of earlier releases

The strategies and assemblers of earlier releases remain available and map
onto an `AssemblyStrategy`:

```text
H2Factory.assemblenearinteractions(op, testspace, trialspace, tree, γ,
    nearstrategy, interaction; verbose=false)
```

| Earlier name | Assembly strategy |
| :--- | :--- |
| `SparseBEASTNearInteractionsAssembler` | `AssemblyStrategy(quadstrat, SparseStorage(), StaticScheduler())` |
| `BlockBEASTNearInteractionsStrategy` | `AssemblyStrategy(quadstrat, BlockSparseStorage(), DynamicScheduler())` |
| `DummyNearInteractionsStrategy` | `AssemblyStrategy(DummyEntries(), SparseStorage(), SerialScheduler())` |

A custom `AbstractNearInteractionsStrategy` implements
`H2Factory.buildnearinteractionsassembler` and returns a subtype of
`H2Factory.AbstractNearInteractionsAssembler{T}`. Its accessors `operator`,
`testspace`, `trialspace` and `quadstrategy` read the fields of these names,
and `assemblystrategy(assembler)` gives its assembly strategy, so any kernel
family with a `buildkernelmatrix` method is assembled. `γ` and the type
parameter `T` no longer set the scalar type: entries have the scalar type of
the kernel matrix.

Next: [storage and multiplication](storage.md).
