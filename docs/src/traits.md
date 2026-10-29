# Traits and dispatch

H2Factory uses small types to select operations through Julia dispatch.
Each family answers a different question. A map orientation, a coupling
format, and a scheduling key are not interchangeable.

## Which choice belongs where?

| Choice | Selected by | Used by |
| :--- | :--- | :--- |
| Map orientation | `H2Factory.maptrait(A)` | Accessors choosing sides and operations |
| Local translation direction | Orientation and traversal phase | Transfer and coupling kernels |
| Overwrite or accumulate | Traversal's contribution order | The kernel writing output |
| I2O representation | `H2Factory.i2otrait(collection)` | `applycouplingkernel!` dispatch |
| I2I representation | `H2Factory.i2itrait(collection)` | Transfer application and scratch allocation |
| Moment storage writer | `H2Factory.storetrait(object)` | `buildstorer` during assembly |
| Scheduling operation | Traversal or local operation | Scheduler lookup |

The [map guide](maps/accessors.md) explains orientation. The
[transfer guide](far/level-transfers.md) explains directions, accumulation,
and combined versus separable transfers. The [coupling guide](far/coupling.md)
separates storage format from geometric reuse.

## Policies and declarations

Some related choices are policies or Boolean queries rather than trait
singletons. `H2Factory.modulation(collection)` returns a conversion policy.
Symmetric near assembly asks whether both the operator and assembler support
that optimization.

None implies another. A Toeplitz storage trait does not mean FFT conversion
has already happened. A mathematically symmetric operator does not imply
that its assembler can construct only one triangle.

See [modulation](far/modulation.md), [symmetric assembly](near/assembly.md#Symmetric-Galerkin-assembly),
and [moment storers](extensions.md#Selecting-a-storer).

## What a new method needs

Built-in workspaces already have a map trait. Built-in transfer and coupling
collections declare their representation traits. Reuse them when their
meaning fits the method.

For a new representation, define the query on your type and implement the
operation selected by its result. Returning `DenseTranslation()` is not
enough to make an arbitrary collection behave like the built-in dense one.
The [required methods](implementing/required.md) distinguish supplied
operations from extension hooks.
