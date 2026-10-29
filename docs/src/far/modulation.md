# Modulation and scratch

The moments used by level transfers are not always in the basis needed by
I2O. A modulation policy supplies that conversion. Coupling and conversion
remain separate so conversion can run once per node, not for every pair.

```text
source moment -> modulate_in! -> pair couplings
received contributions -> modulate_out! -> level/leaf operations
```

## Pair the policy with its coupling

```julia
translation = H2Factory.I2OTranslation(coupling, policy)
```

The one-argument constructor reads `H2Factory.modulation(coupling)`.
Ordinary collections default to `NoModulation()`; Toeplitz collections use
`ToeplitzModulation()`. A wrapper can forward its inner policy explicitly.

Three choices meet here: the coupling trait selects its kernel, the moment
type supplies storage, and modulation supplies conversion. Selecting
`ToeplitzI2OTranslationTrait()` does not perform an FFT; traversal must still
apply the matching modulation around the pair couplings.

For `MyPolicy <: H2Factory.AbstractI2OModulation`, implement:

```text
H2Factory.modulate_in!(moment, level::Int, maptrait, policy::MyPolicy,
    coupling, channel, scheduler, buffer)
H2Factory.modulate_out!(moment, level::Int, maptrait, policy::MyPolicy,
    coupling, channel, scheduler, buffer)
```

Both modify the moment in place and return it. The output operation is the
conversion needed by the factorization; it is not necessarily a mathematical
inverse of the input operation. Respect the map orientation in both.

## Allocate scratch per workspace

`H2Factory.buildi2ochannel(core)` returns the workspace's I2O scratch.
`H2Factory.i2ochannels(coupling, translatingplan)` can help construct it;
its default is `EmptyChannels()` for collections needing none.

Modulation receives the channel and its borrowed buffer. Coupling kernels
receive local scheduling and borrowed scratch through `TranslationContext`.
Read-only conversion matrices can be core data; mutable conversion buffers
and FFT plans cannot be shared between concurrent workspaces.

See [chunk allocation](../maps/scheduling.md) for sizing and borrowing rules.
