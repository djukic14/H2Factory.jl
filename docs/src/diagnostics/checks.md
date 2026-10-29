# Structural checks

`H2Factory.Diagnostics` inspects a constructed map. It checks relationships
between plans, moments, and execution data without assembling a dense reference.

```julia
using H2Factory.Diagnostics

report = sanitycheck(A)
```

`haserrors(report)` reports error-severity findings. The report is useful for
finding a broken construction contract, not for proving approximation accuracy.

## Focused checks

| Call | Examines |
| :--- | :--- |
| `H2Factory.Diagnostics.checkplans(A)` | Plan structure and relationships |
| `H2Factory.Diagnostics.checkmoments(A)` | Moment layout and expected data |
| `H2Factory.checkmomentaliasing(A)` | Pool ranges, overlap, and stored-moment assumptions |
| `H2Factory.checkexecutiondata(A)` | Schedules and execution data |
| `H2Factory.Diagnostics.admissibilityreport(A)` | Geometry of selected near/far interactions |

`checkmomentaliasing(A).ok` reports whether its layout checks passed. The
overlap scan can be quadratic in the number of moments; run it during
construction or tests, not for every product.

When a check fails, inspect the relevant [allocation](../moments/allocation.md),
[partition](../near/assembly.md), or [scheduling](../maps/scheduling.md) contract.
For numerical validation, use the [implementation tests](../implementing/testing.md).
