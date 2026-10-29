# Near storage and multiplication

The core exposes the assembled near operator through
`H2Factory.nearinteractions(core, H2Factory.isForward())`.
Its default reads `core.nearinteractions`.

## Storage choices

| Representation | Use |
| :--- | :--- |
| Dense matrix | Simple small examples or an externally computed operator |
| `SparseMatrixCSC` | Individual near entries in global row/column positions |
| Block-sparse matrix | Dense near blocks with row/column index groups |
| Custom matrix-like object | Your own representation with the required `mul!` methods |

The full matrix size must match the test and trial spaces, even though only
near entries contribute. Petrov operators can be rectangular.

`H2Factory.SparseBEASTNearInteractionsAssembler{T}` uses the H2BEAST extension.
`H2Factory.BlockBEASTNearInteractionsAssembler{T}` uses H2BlockSparse.
These select assembly and storage, not the geometric near predicate.
Symmetric assembly requires the [explicit declarations](assembly.md); symmetry
here means transpose symmetry, not automatically Hermitian symmetry.

## Apply the near operator

```text
H2Factory.nearmul!(y, A, x)
H2Factory.nearmul!(y, A, x, α, β)
```

The second computes `y = α * N * x + β * y`, where N is the oriented near
operator. The generic implementation delegates to `LinearAlgebra.mul!`.
A custom storage type can therefore implement matrix multiplication rather
than replace the whole H2 traversal.

For `transpose(A)` or `adjoint(A)`, the accessor returns the corresponding
view of N. The storage must support the resulting products as well.

## Extract a reference partition

`H2Factory.nearinteractions(F, A)` extracts near entries from an independent
matrix-like reference F using A's geometry and orientation. This overload is
different from the one-argument accessor `nearinteractions(A)`.

Use it for checks, not as proof that a supplied near matrix was assembled
correctly. The [testing guide](../implementing/testing.md) describes independent
references and near/far comparisons.
