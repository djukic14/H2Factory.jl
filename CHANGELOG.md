# Changelog

All notable changes to H2Factory are recorded here.

This file starts at the `refactor/interfaces` work and is not a retroactive reconstruction of the
project's earlier history. Entries below cover everything on that branch; anything released before it
is not listed.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This package is
pre-1.0, so breaking changes may land in a minor release.

## [Unreleased]

### Breaking

#### H2Trees is no longer re-exported

H2Factory used to re-export the `H2Trees` module binding together with eleven names from it:
`TwoNTree`, `SimpleHybridTree`, `buildtree`, `buildforest`, `TwoNTreeBuilder`, `BlockTreeBuilder`,
`BoundingBallTreeBuilder`, `KMeansTreeBuilder`, `MetisTreeBuilder`, `MetisForestBuilder` and
`SimpleHybridTreeBuilder`. It no longer exports any of them.

Code that said only `using H2Factory` and then reached for a tree type or a builder will now fail
with an `UndefVarError`. **Add a direct import:**

```julia
using H2Factory
using H2Trees          # <- add this

tree = H2Trees.buildtree(points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.1))
```

Add `H2Trees` to your `[deps]` as well; relying on it arriving through H2Factory was always
incidental.

Note the sharp edge if you prefer importing names rather than the module: **`import H2Trees: TwoNTree`
does NOT bind the module name `H2Trees`.** Only `import H2Trees` or `using H2Trees` does. If any call
site in the same module writes a qualified `H2Trees.something`, the selective-import form will still
fail there. When in doubt use `using H2Trees`.

#### The adjoint leaf multiply must be declared

`leafmul` no longer has a shared default covering all three orientations. Forward and transpose keep
one; the ADJOINT does not, and a map that has not declared it now gets an error naming the type.

The adjoint is `conj(α)·conj(L_trial)ᵀ·Tᴴ·conj(L_test)`, so it conjugates BOTH leaf collections on
top of the transpose the coupling already applies. The plain matvec is correct only when those
collections are real, and no generic can tell: a map's `eltype` does not answer it, since a
`ComplexF64` map can hold `Float64` leaf blocks. A map with COMPLEX leaf collections that never wrote
an adjoint method previously got a well-formed, believable, wrong field.

If your map's leaf moment collections are real, declare the plain matvec explicitly:

```julia
H2Factory.leafmul(::MyCore, ::H2Factory.isAdjoint) = H2Factory.LinAlgebraMulFunctor()
```

If they are complex, supply a conjugating functor instead.

#### Other breaking changes

- The interaction-strategy accessor `isnear(strategy)` is renamed to
  `nearcriterion(strategy)`. The `InteractionStrategy(; isnear=...)` keyword is unchanged.
- `AbstractNearFieldStrategy` and `NearFieldStrategy` are renamed to
  `AbstractInteractionStrategy` and `InteractionStrategy`. The strategy controls both the near/far
  partition and the traversal plan, not only near-field assembly.
- `store!` is renamed to `buildstorer`. The old name is gone rather than deprecated.
- The two transfer hooks are promoted out of the private namespace into the declared protocol.
  Code reaching for them under their old private names must move to the protocol names.
- The universal `conjmul!` fallback is removed. `conjmul!` still exists, but a type that relied on
  the catch-all now has to declare the behaviour explicitly; moment type generation and behaviour
  registration are separate steps. Previously a type could silently inherit a fallback that was
  wrong for it.
- The duplicate I2O protocol is collapsed onto one guarded entry point and one kernel,
  `applyi2o!` / `applycoupling!`. Extensions implementing the retired second spelling must move.
- The empty `assemble` fallbacks are removed. An incomplete assembler used to satisfy the interface
  silently and do nothing; it now fails loudly.
- `Match`, `MacroTools` and `OrderedCollections` are dropped from `[deps]`. None were used. This
  only affects you if you were relying on H2Factory to pull one of them into your environment.
- The package version is dropped to `0.1.0-DEV` so consumers bound it deliberately.
- The memory-reporting API (`Diagnostics.memoryreport`, `memoryestimate`,
  `StorageReport`, and `StorageEstimate`) is removed. `H2Summary` now contains only
  map and tree statistics; its `storage` and `estimate` fields are removed.

### Added

- `H2Factory.Interfaces`, an explicit namespace declaring the package's extension protocol. The
  near-interaction assembly, traversal optimization, I2O, leaf multiplication, moment storage,
  interaction strategy and workspace construction hooks are all declared there rather than being
  implied by whichever method an extension happened to find.

### Fixed

- Block-sparse near assembly supports BEAST methods without the optional `primer` keyword.
- The test environment's `[sources]` no longer pin absolute machine-specific paths, so the suite
  resolves outside one developer's home directory.
- The generated `getindex` example for moments is runnable. It constructed the ABSTRACT type
  (`AbstractMoment{2,Float64}((3, 3))`, which has no constructor) and asserted
  `m[1] == m.moments[1]`, which is not reliably true because a fresh moment's storage is
  uninitialized and `NaN == NaN` is false. It now names the concrete, qualified
  `H2Factory.Moment` and uses `===`. A test evaluates every rendered example's constructor, so
  neither shape can return silently.
