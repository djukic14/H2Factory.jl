# Toeplitz coupling

When the method's grid and kernel produce Toeplitz structure, I2O can use FFTs
instead of a dense coupling matrix. H2Factory supplies storage and application;
the method must establish that this representation is valid and build its
Fourier-domain coefficients.

## Read-only data and workspace plans

`H2Factory.ToeplitzI2OTranslationCollection(collection, tree)` wraps the stored
couplings and the tree needed to resolve levels. It does not own mutable FFT plans.

The H2Toeplitz extension provides the read-only builder:

```text
H2Factory.toeplitzi2ocollection(tree, translatingplan, reuse, builder; kwargs...)
```

The convenience tree constructor `ToeplitzI2OTranslationCollection(...)`
returns both the collection and a scratch channel. For a core/workspace split,
build the read-only collection once and allocate channels per workspace.

```julia
translation = H2Factory.I2OTranslation(coupling)
channel = H2Factory.i2ochannels(translation, translatingplan)
```

Run channel construction inside the workspace's allocation scope: it uses
`H2Factory.nchunks(level)`. The resulting `ToeplitzI2OChannel` holds FFT plan
pools by level and chunk. A concrete FFT backend, such as FFTW, is needed in
addition to AbstractFFTs.

## Conversion is explicit

The collection defaults to `H2Factory.ToeplitzModulation()`. It converts
[Toeplitz moments](../moments/toeplitz.md) into and out of the coupling basis.
Pairing a bare Toeplitz collection with `NoModulation()` errors. The same
policy is rejected for a Toeplitz moment even through a collection wrapper.

A wrapper must forward its modulation or provide its own. Padded storage
alone does not prove that an FFT has been performed.

Transpose and adjoint use different plan/coefficient handling. The built-in
channel and kernels implement that routing; custom ones must preserve it.

Toeplitz and [symmetry reduction](symmetry.md) are separate choices. A method
may use either, both, or neither.

