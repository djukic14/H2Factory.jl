# Moment pools and aliasing

[API reference](../apiref.md) | [Storage guide](../moments/overview.md)

Use `storagemomentsplan` to allocate node moments and `checkmomentaliasing` to
inspect a map's layout. The lower-level allocation and pointer helpers are also
listed here because their source docstrings explain that layout. They are
implementation details, not an alternative ownership contract for custom arrays.

```@autodocs
Modules = [H2Factory]
Order = [:function]
Filter = object -> object in (
    H2Factory.storagemomentsplan, H2Factory.allocatemomentbuffers,
    H2Factory.planmomentstorage, H2Factory.materializemoments,
    H2Factory.checkmomentaliasing, H2Factory.momentextents, H2Factory.checkextents,
    H2Factory.checktranslatingnodesstored,
    H2Factory.alignmentslack, H2Factory.wrapbuffer,
)
```
