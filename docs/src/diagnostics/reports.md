# Structure and timing reports

Start with `H2Factory.Diagnostics.h2summary(A)` for a compact overview.
For Petrov maps, its tree statistics describe the test tree.

## Structure

| Call in `H2Factory.Diagnostics` | Meaning |
| :--- | :--- |
| `h2summary(A)` | Map size, tree statistics, leaf occupancy, and far-pair counts |
| `translation_counts(A)` | Coupling construction and reuse information where available |

## Timing

```julia
using H2Factory.Diagnostics
timing = profilemul(A, x)
rows = diagnostic_rows(timing)
```

The call executes a multiplication; `timing.y` is its result. Records can
include whole operations and per-level phases. Nested timings overlap, so
their sum is not necessarily the total time. Compilation and warmup also
matter when comparing runs.

Use timings to compare warmed-up runs of the same configuration.
