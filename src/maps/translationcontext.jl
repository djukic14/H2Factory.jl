# Explicit translation direction, accumulation, and execution-context types. Direction is applied at
# both coupling and level-transfer boundaries; kernels receive scheduling, scratch, and diagnostics
# through `TranslationContext`.

abstract type AbstractTranslationDirection end

"""
Apply the translation as stored.
"""
struct ForwardTranslation <: AbstractTranslationDirection end

"""
Apply `transpose(T)` at the coupling or level-transfer boundary, without conjugating moments.
"""
struct TransposeTranslation <: AbstractTranslationDirection end

"""
Apply `adjoint(T)` at the coupling or level-transfer boundary.

This remains distinct from [`TransposeTranslation`](@ref), even when a real transfer makes the two
operations numerically equal.
"""
struct AdjointTranslation <: AbstractTranslationDirection end

"""
Apply `conj(T)` without transposing it.

This direction is used by adjoint aggregation. For
`A = α · (I2I_test · L_test)^T · T_i2o · (I2I_trial · L_trial)`, the adjoint is

    A^H = conj(α) · (conj(I2I_trial) · conj(L_trial))^T · T_i2o^H · conj(I2I_test) · conj(L_test)

so aggregation remains child-to-parent while conjugating the transfer. For real transfers this is
equivalent to [`ForwardTranslation`](@ref).
"""
struct ConjugateTranslation <: AbstractTranslationDirection end

abstract type AbstractAccumulationMode end

"""
Overwrite the target moment with the translation result.
"""
struct Overwrite <: AbstractAccumulationMode end

"""
Accumulate the translation result into the target moment.
"""
struct Accumulate <: AbstractAccumulationMode end

# Translation direction corresponding to each map orientation.
translationdirection(::isForward) = ForwardTranslation()
translationdirection(::isTranspose) = TransposeTranslation()
translationdirection(::isAdjoint) = AdjointTranslation()

# Aggregation always runs child-to-parent. Forward and transpose maps therefore use the stored
# transfer, while adjoint maps conjugate it; see `ConjugateTranslation`.
aggregationdirection(::isForward) = ForwardTranslation()
aggregationdirection(::isTranspose) = ForwardTranslation()
aggregationdirection(::isAdjoint) = ConjugateTranslation()

abstract type AbstractDiagnostics end

"""
Diagnostics policy that records nothing.
"""
struct NoDiagnostics <: AbstractDiagnostics end

record!(::NoDiagnostics, args...; kwargs...) = nothing

# Keep disabled timing equivalent to calling `f()` directly.
@inline with_timing(f, ::NoDiagnostics, args...; kwargs...) = f()

# Map orientation label used by diagnostics.
diagnosticmode(A) = diagnosticmode(maptrait(A))
diagnosticmode(::isForward) = :forward
diagnosticmode(::isTranspose) = :transpose
diagnosticmode(::isAdjoint) = :adjoint

"""
    TranslationContext(scheduler, buffer[, diagnostics])

Execution context for a local translation kernel: its scheduler, borrowed scratch buffer, and
diagnostics policy. Diagnostics default to `NoDiagnostics()`.
"""
struct TranslationContext{S,B,D<:AbstractDiagnostics}
    scheduler::S
    buffer::B
    diagnostics::D
end
function TranslationContext(scheduler, buffer)
    return TranslationContext(scheduler, buffer, NoDiagnostics())
end

# --- Level transfers (I2I / O2O) ---------------------------------------------------------------
#
# Level transfers use `leveltranslation!`, not the I2O coupling protocol. The selected direction is
# applied directly to the sector-indexed transfer.
