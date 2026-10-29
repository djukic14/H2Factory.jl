# Symmetry reduction

Translation reuse has several levels. Equal displacements can share a
coupling. Symmetry reduction goes further: displacements related by a
declared lattice symmetry can share one canonical coupling.

H2Trees finds the geometry and symmetry tags. H2Factory applies the matching
sample permutations and, where needed, component transformations.

## Choose a representation

| Constructor in H2Factory | Representation |
| :--- | :--- |
| `symmetryreducedcollection(canonical, permutationfor)` | Sample-permuted couplings, such as angular samples |
| `symmetryreduceddensecollection(canonical, gridsizefor)` | Dense couplings with row/column grid permutations |
| `symmetryreducedtoeplitzcollection(canonical; componentaction=...)` | Permuted Toeplitz difference tensors |

`canonical` must carry tags, usually obtained with
`H2Trees.SymmetryDirectionInvariancePerLevel`. Merely computing fewer blocks
and discarding the tags would lose how each actual pair relates to them.

For dense coupling, grid shapes belong to receiver rows and source columns.
The three-argument form accepts separate callbacks for rectangular grids.
Grid points must be closed under the declared symmetry. The method also
verifies that the physical kernel obeys that symmetry law.

## Components matter too

Scalar kernels need only a sample permutation. Vector and tensor kernels
can also need component swaps or signs. H2Factory distinguishes
`VectorField()`, `TrueTensor()`, and `PseudoTensor()`; these are not inferred
from the shape of a matrix container.

The dense reduction supports scalar and true/pseudo-tensor couplings, not
`VectorField()`. Other representations have their own supported actions.
Reduction introduces no further approximation only when the method's grid
and kernel satisfy the declared rules.

## Guard the unreduced path

A canonical collection with unconsumed tags cannot use the ordinary kernel
silently. `H2Factory.assertnosymmetry` guards that boundary. Reduced wrappers
account for the tags in their values, then allow normal application.

The reduction constructors do not provide a universal fallback for arbitrary
geometry. H2Trees distinguishes unsupported tree types from
`NonLatticeTranslationError` for incompatible root offsets. A method can
deliberately fall back for the latter; catching every error would hide defects.

See [H2Trees translations](https://djukic14.github.io/H2Trees.jl/dev/translations/)
for the geometric sharing policies.
