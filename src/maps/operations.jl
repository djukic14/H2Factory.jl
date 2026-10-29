# Operation tags separate outer traversal scheduling from scheduling inside local kernels.
#
# Traversal phases: `AggregateOp`, `I2OOp`, and `DisaggregateOp`.
# Inner kernels: `I2IOp`, `I2OOp`, `O2OOp`, `ModulationOp`, `ReverseModulationOp`, and `LeafMulOp`.
# Plan tags below identify the orientation-dependent node set used by an outer phase.
abstract type AbstractH2Operation end

struct AggregateOp <: AbstractH2Operation end
struct I2IOp <: AbstractH2Operation end
struct I2OOp <: AbstractH2Operation end
struct O2OOp <: AbstractH2Operation end
struct DisaggregateOp <: AbstractH2Operation end
struct ModulationOp <: AbstractH2Operation end
struct ReverseModulationOp <: AbstractH2Operation end
struct LeafMulOp <: AbstractH2Operation end

# Plan tags identify the concrete node set walked by an outer loop. `outerscheduler` resolves the
# traversal phase and map orientation to one of these plans:
#
#     forward   AggregateOp    -> TrialAggregationOp        transpose/adjoint -> TestAggregationOp
#     forward   DisaggregateOp -> TestDisaggregationOp      transpose/adjoint -> TrialDisaggregationOp
struct TrialAggregationOp <: AbstractH2Operation end
struct TestAggregationOp <: AbstractH2Operation end
struct TestDisaggregationOp <: AbstractH2Operation end
struct TrialDisaggregationOp <: AbstractH2Operation end

"""
    scheduledop(op, maptrait) -> AbstractH2Operation

Return the inner-scheduler key for this map orientation.
Forward preserves the key. Transpose/adjoint swap `I2IOp` and `O2OOp`, keeping each
scheduler with its tree:

    innerscheduler(A, O2OOp(), level) == innerscheduler(transpose(A), I2IOp(), level)

Other keys stay unchanged. This changes scheduler lookup, not the kernel executed.
Outer phases select their plans separately.
"""
scheduledop(op::AbstractH2Operation, ::isForward) = op
scheduledop(op::AbstractH2Operation, ::Union{isTranspose,isAdjoint}) = op
scheduledop(::I2IOp, ::Union{isTranspose,isAdjoint}) = O2OOp()
scheduledop(::O2OOp, ::Union{isTranspose,isAdjoint}) = I2IOp()

# Scheduling containers for these tags are defined in `maps/scheduling.jl`.
