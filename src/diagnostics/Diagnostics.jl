"""
    H2Factory.Diagnostics

Inspect H2 maps without changing them, or collect multiplication timings.

Start with [`h2summary`](@ref) and [`sanitycheck`](@ref).

[`translation_counts`](@ref) reports coupling reuse. [`profilemul`](@ref) and the
`timed*!` functions collect timings using the normal multiplication path.
"""
module Diagnostics

using LinearAlgebra
using ..H2Factory:
    H2Factory,
    AbstractH2Map,
    AbstractH2Maps,
    maptrait,
    isForward,
    isTranspose,
    isAdjoint,
    nearinteractions,
    nearmul!,
    farmul!,
    i2otranslator,
    coupling,
    aggregationplan,
    disaggregationplan,
    testmoments,
    trialmoments,
    checkmomentaliasing,
    forwardmap,
    testaggregationplan,
    trialaggregationplan,
    testdisaggregationplan,
    trialdisaggregationplan
# Extend the diagnostics protocol declared by the parent module.
import ..H2Factory: AbstractDiagnostics, NoDiagnostics, record!, with_timing
using ..H2Factory: diagnosticmode
using H2Trees

include("timing.jl")
include("structure.jl")
include("show.jl")

# Timing
export AbstractDiagnostics, NoDiagnostics, TimingDiagnostics, TimingRecord, TimingReport
export profilemul, timedmul!, timednearmul!, timedfarmul!
export record!, with_timing, totalseconds, diagnostic_rows

# Structure
export h2summary, translation_counts
export H2Summary, TranslationCounts

# Structural checks
export sanitycheck, checkplans, checkmoments
export admissibilityreport
export SanityReport, DiagnosticIssue, haserrors

end # module Diagnostics
