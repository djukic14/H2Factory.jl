# Workspaces and scheduling

[API reference](../apiref.md) | [Scratch and scheduling](../maps/scheduling.md)

`H2Workspace` builds an executable map; `similarworkspace` allocates another
workspace over the same core. `ExecutionScheduling` selects per-phase and
per-level settings. Low-level helpers are included below for reference.

```@autodocs
Modules = [H2Factory]
Order = [:function]
Filter = object -> object in (
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
)
```
