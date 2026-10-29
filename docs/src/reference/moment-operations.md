# Conjugated and diagonal multiplication

[API reference](../apiref.md) | [Translation guide](../far/level-transfers.md)

`conjmul!` handles conjugated moment-collection multiplication for linked pairs.
`elementwisemul!` applies diagonal couplings. Ordinary matrix multiplication has
its own [reference page](multiplication.md).

```@autodocs
Modules = [H2Factory]
Order = [:function]
Filter = object -> object in (H2Factory.conjmul!, H2Factory.elementwisemul!)
```
