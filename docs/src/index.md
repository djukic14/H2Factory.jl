# H2Factory

H2Factory runs the matrix-vector product for a hierarchical fast method.
It walks the trees, applies near and far interactions, and manages moments
and scratch. Your method supplies the numerical data and operations.

## Two ways into the guide

[Traits and dispatch](traits.md) explains how the small types and policies
select operations throughout the components below.

**To understand the code**, start with [How the multiplication works](algorithm.md).
Then follow the components below.

**To implement a method**, start with [Required methods](implementing/required.md).
The [worked example](implementing/example.md) connects those methods into a
small complete map. It needs no application-scale assembly.

[Construction and execution](construction.md) connects method construction,
workspace allocation, and multiplication.

| Component | What it covers |
| :--- | :--- |
| [Maps](maps/core-workspace.md) | Shared core data, workspaces, accessors, and scheduling |
| [Moments](moments/overview.md) | Leaf collections, moment types, allocation, and storage reuse |
| [Near interactions](near/assembly.md) | Selecting, computing, and storing direct interactions |
| [Far interactions](far/level-transfers.md) | Level transfers, I2O coupling, modulation, and traversal |
| [Diagnostics](diagnostics/checks.md) | Structural checks, map summaries, and timings |

The [API reference](apiref.md) provides individual signatures. Most extension
hooks are qualified, for example `H2Factory.buildmoments`. Tree functions come
from `H2Trees`, imported separately.

H2Factory does not choose a physical kernel, interpolation basis, or quadrature
rule for every method. Those choices belong to the implementation. Optional
[extensions](extensions.md) provide BEAST assembly and FFT-based coupling.
