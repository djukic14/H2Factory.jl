# Assembly extensions

The core package does not require a particular boundary-element package or
FFT backend. Extensions add those operations when their dependencies load.

| Extension | Dependencies | Provides |
| :--- | :--- | :--- |
| `H2BEAST` | BEAST, ClusterTrees, CompScienceMeshes | Moment assembly, sparse near assembly, operator wrappers |
| `H2BlockSparse` | BEAST, CompScienceMeshes, BlockSparseMatrices | Block-sparse near assembly |
| `H2Toeplitz` | AbstractFFTs | Toeplitz coupling construction and application |

A concrete FFT backend, such as FFTW, must also provide the planning operations.
Declaring a dependency does not load it; `Base.get_extension(H2Factory, :H2BEAST)`
returns the extension module only after it has loaded.

## Moment assembly and storers

`H2Factory.assemblemoments!` supplies the BEAST integration route. The method
chooses the expansion integrand and quadrature. A storer receives each assembled
contribution and writes it to the leaf collections.

```text
H2Factory.storetrait(::Type{MyAssembler})
H2Factory.buildstorer(F, assembler, storeinfo)
H2Factory.buildstorer(Fs, assembler, storeinfo, tree)
(storer)(value, cluster, m, n)
```

The default builders construct the type returned by `storetrait`, with
`(F, storeinfo)` or `(Fs, storeinfo, tree)`.
`H2Factory.StoreAsIs` adds values directly, mapping global indices to leaf
positions when a tree is supplied. The object passed to `buildstorer` must
have the matching `storetrait` method; it is not inferred from arbitrary arrays.

A method computing its collections analytically can skip this assembly route
and expose finished collections on the core.

### Selecting a storer

`H2Factory.storetrait(object)` delegates to `storetrait(typeof(object))`.
Its result is a **type**, such as `H2Factory.StoreAsIs`, not an instance
such as `StoreAsIs()`. `buildstorer` constructs it with the destination
arrays and assembly metadata.

This example uses the flat-array form. The storer adds a component value
at row 1, column 2; it needs no BEAST integration or tree:

```jldoctest storer-trait-guide
using H2Factory

struct GuideStorage end
H2Factory.storetrait(::Type{GuideStorage}) = H2Factory.StoreAsIs
arrays = [zeros(2, 2)]
writer = H2Factory.buildstorer(arrays, GuideStorage(), nothing)
writer([3.0], nothing, 1, 2)
arrays[1][1, 2] == 3.0

# output

true
```

The tree form also maps global basis indices to positions in each leaf.
A custom writer supplies the corresponding constructors and callable
`(value, cluster, m, n)` method. It controls storage of assembled
contributions, not how moments are allocated for multiplication.

## Operator wrappers

H2BEAST supplies macros such as `H2Factory.@efie` and
`H2Factory.@mw3Ddoublelayer` to define wrapper types. Their coefficients are
part of their factorization. The EFIE wrapper requires
`alpha = beta * gamma^2`; inconsistent input raises `ArgumentError`.

These wrappers do not by themselves implement a complete fast method.
The method still supplies moments, translations, and workspace builders.

See [near assembly](near/assembly.md), [collections](moments/overview.md), and
[Toeplitz coupling](far/toeplitz.md) for the relevant execution contracts.
