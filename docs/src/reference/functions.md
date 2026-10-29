# Functions

[API reference](../apiref.md) | [Implementation guide](../implementing/required.md)

Core accessors and builder results are listed in the
[implementation checklist](../implementing/required.md).
Workspace helpers, moment pools, multiplication, extension declarations, and
diagnostics have separate reference pages.

```@autodocs
Modules = [H2Factory]
Order = [:function]
Filter = object -> !(object in (
    H2Factory.LinearAlgebra.mul!, H2Factory.conjmul!, H2Factory.elementwisemul!,
    H2Factory.storagemomentsplan, H2Factory.allocatemomentbuffers,
    H2Factory.planmomentstorage, H2Factory.materializemoments,
    H2Factory.checkmomentaliasing, H2Factory.momentextents, H2Factory.checkextents,
    H2Factory.checktranslatingnodesstored,
    H2Factory.alignmentslack, H2Factory.wrapbuffer,
    H2Factory.H2Workspace, H2Factory.similarworkspace, H2Factory.core, H2Factory.plans,
    H2Factory.workspacebuffers, H2Factory.scheduling, H2Factory.schedules,
    H2Factory.buildleafchannels, H2Factory.momentbuffers, H2Factory.busyflag,
    H2Factory.nchunks, H2Factory.maxnchunks, H2Factory.withnchunks,
    H2Factory.withschedulechunks, H2Factory.schedulechunkcounts, H2Factory.requestedntasks,
    H2Factory.executionscheduling, H2Factory.defaultexecutionscheduling,
    H2Factory.serialexecutionscheduling, H2Factory.serialinnerscheduling,
    H2Factory.perlevel, H2Factory.outerscheduling, H2Factory.outerscheduler,
    H2Factory.innerscheduler, H2Factory.operationscheduler, H2Factory.scheduledop,
    H2Factory.checkexecutiondata, H2Factory.createchannel, H2Factory.borrowbuffer,
    H2Factory.acquireworkspace!, H2Factory.releaseworkspace!, H2Factory.withworkspace,
))
```
