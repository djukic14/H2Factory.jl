# Optional capabilities

Start with a consistent forward map. Add capabilities where the method has
the corresponding numerical operations, rather than inheriting a plausible
but incorrect default.

## Transpose and adjoint

Collections, transfers, near storage, and I2O kernels must all support the
requested orientation. Adjoint also needs conjugation at the leaves and in
the far-field prefactor. [Accessors and orientation](../maps/accessors.md)
shows the declarations. Check against an independently oriented reference.

## Different test and trial sides

Use `PetrovGalerkinH2Core{T}`. Supply separate leaf collections and all four
plans. Side-specific translators can differ. Return vector buffers in
`H2Factory.VectorBuffers(test, trial)` and transfer channels in
`H2Factory.SidedChannels(test, trial)` when they are side-specific.

The [named allocation form](../moments/allocation.md) prevents accidental
test/trial callback swaps. H2Factory does not force a particular polynomial
family or grid compatibility rule on every Petrov method.

## Accelerated representations

[Toeplitz](../far/toeplitz.md) requires a valid coupling representation,
compatible moment storage, modulation, and workspace FFT plans.
[Symmetry](../far/symmetry.md) requires the geometric tags and the correct
sample and component transformations. Neither is implied by the other.

## Threading

Read-only core data can be shared. Mutable scratch must be per workspace
and, where operations overlap, per chunk. [Scheduling](../maps/scheduling.md)
explains allocation counts. Test more than one chunk count; a serial success
does not exercise the whole buffer layout.

## Custom traversal hooks

The protocol includes hooks such as `H2Factory.aggregateleafchunk!` for
specialized leaf work. These are for implementations needing a different
local execution path, not prerequisites for a basic map.
Preserve initialization, stored-moment liveness, and workspace ownership.

Keep a small [independent reference test](testing.md) as capabilities are added.
