# Testing an implementation

A map can load and multiply while still using the wrong side, dropping an
interaction, or reusing a live buffer. Test the assembled operation as well
as its individual kernels.

## Compare against an independent reference

For a small problem, assemble a full reference F without using the H2 map's
own multiplication. Compare near and far parts as well as their sum.
`H2Factory.nearinteractions(F, A)` and `H2Factory.farinteractions(F, A)` use A's
partition to extract the corresponding reference entries.

[H2FactoryTestUtils](https://github.com/djukic14/H2FactoryTestUtils.jl#readme)
provides assertion-based helpers:

| Call | Checks |
| :--- | :--- |
| `checkmul(F, A)` | Product forms, including scaled `mul!` |
| `checkinteractions(F, A)` | Partition and near/far/full products |
| `checknearmul(Fnear, A)` | Near contribution |
| `checkfarmul(Ffar, A)` | Far contribution |
| `checki2otranslation(...)` | One coupling, including its basis conversions |

For transpose or adjoint, orient both the reference and map. Real input alone
can miss conjugation mistakes, so include complex data where supported.

## Test contracts, not only answers

- Test overwrite with a nonzero destination, then accumulation.
- Exercise each supported direction through its public entry point.
- Run with different chunk counts and independent workspaces.
- Include unequal test/trial sizes when supporting Petrov maps.
- Check that an intentionally unsupported configuration is actually rejected.

Sampled product checks are not bounds for every input. Choose tolerances
from the approximation being tested, not from a single fortunate vector.
Comparison with another implementation checks agreement, not independent truth.

## Use small synthetic maps

H2FactoryTestUtils offers `fakegalerkinmap`, `fakepetrovmap`,
`faketoeplitzmap`, and `fakepetrovtoeplitzmap`. Its `denseh2matrix` reference
describes synthetic far interactions; it is not a full physical BEM matrix.

The [worked example](example.md) needs no TestUtils dependency and checks
all three orientations on a small kernel. Keep costly application assembly
out of normal documentation builds.

Structural checks belong in [Diagnostics](../diagnostics/checks.md). They
complement numerical tests; they do not replace them.
