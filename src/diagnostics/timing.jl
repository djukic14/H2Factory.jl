# Timing uses the same multiplication path as normal execution. `NoDiagnostics` records nothing;
# `TimingDiagnostics` collects records from that path.

"""
    TimingRecord

One timed operation. `operation` names the work, `phase` its scope, and `mode` the map orientation.
`level` and `node` are `missing` when they do not apply. Allocation data is only reliable for
whole-operation records.
"""
struct TimingRecord
    operation::Symbol
    phase::Symbol
    level::Union{Int,Missing}
    node::Union{Int,Missing}
    mode::Symbol
    seconds::Float64
    # Reserved until allocation counts can be measured reliably.
    allocations::Int
    bytes::Int
    count::Int
end

"""
    TimingDiagnostics(; bylevel=true, bynode=false)

Collect [`TimingRecord`](@ref)s. Set `bylevel=false` to keep only whole-operation records.
`bynode` is reserved for future per-node timing.
"""
mutable struct TimingDiagnostics <: AbstractDiagnostics
    records::Vector{TimingRecord}
    bylevel::Bool
    bynode::Bool
end
TimingDiagnostics(; bylevel::Bool=true, bynode::Bool=false) =
    TimingDiagnostics(TimingRecord[], bylevel, bynode)

# Recording methods for `TimingDiagnostics`.

function record!(
    d::TimingDiagnostics,
    operation::Symbol,
    phase::Symbol;
    level=missing,
    node=missing,
    mode::Symbol=:forward,
    seconds::Float64=0.0,
    allocations::Int=0,
    bytes::Int=0,
    count::Int=1,
)
    push!(
        d.records,
        TimingRecord(
            operation, phase, level, node, mode, seconds, allocations, bytes, count
        ),
    )
    return nothing
end

function with_timing(
    f,
    d::TimingDiagnostics,
    operation::Symbol,
    phase::Symbol;
    level=missing,
    node=missing,
    mode::Symbol=:forward,
    count::Int=1,
)
    # Disabled scopes still run their work; they only skip the record.
    (phase === :level && !d.bylevel) && return f()
    (phase === :node && !d.bynode) && return f()

    stats = @timed f()
    record!(
        d,
        operation,
        phase;
        level=level,
        node=node,
        mode=mode,
        seconds=stats.time,
        bytes=stats.bytes,
        count=count,
    )
    return stats.value
end

"""
    TimingReport(y, records, metadata)

Result of a profiled multiplication. Contains the result `y`, timing records, and map metadata.
"""
struct TimingReport{Y,R,M}
    y::Y
    records::R
    metadata::M
end

function _timingmetadata(A)
    return (;
        maptype=nameof(typeof(A)),
        size=size(A),
        mode=diagnosticmode(A),
        nthreads=Threads.nthreads(),
        treetype=nameof(typeof(H2Trees.tree(A))),
    )
end

"""
    timedmul!(y, A, x; diagnostics=TimingDiagnostics()) -> y

Multiply `A * x` into `y` and collect timing records. The result is the same as for `mul!`.
"""
function timedmul!(y, A, x; diagnostics::AbstractDiagnostics=TimingDiagnostics())
    with_timing(diagnostics, :mul, :total; mode=diagnosticmode(A)) do
        return H2Factory._unsafe_mul!(y, A, x, diagnostics)
    end
    return y
end

"""
    timednearmul!(y, A, x; diagnostics=TimingDiagnostics()) -> y

Near-field pass only, timed as `:near`/`:total`.
"""
function timednearmul!(y, A, x; diagnostics::AbstractDiagnostics=TimingDiagnostics())
    with_timing(diagnostics, :near, :total; mode=diagnosticmode(A)) do
        return H2Factory.nearmul!(y, A, x)
    end
    return y
end

"""
    timedfarmul!(y, A, x; diagnostics=TimingDiagnostics()) -> y

Run only the far-field pass and overwrite `y` with its result. Records the total and, when enabled,
each far-field level.
"""
function timedfarmul!(y, A, x; diagnostics::AbstractDiagnostics=TimingDiagnostics())
    T = eltype(A)
    with_timing(diagnostics, :far, :total; mode=diagnosticmode(A)) do
        return farmul!(y, A, x, one(T), zero(T), diagnostics)
    end
    return y
end

"""
    profilemul(A, x; bylevel=true, bynode=false) -> TimingReport

Profile `A * x` and return its result with the collected timings. Set `bylevel=false` to omit
per-level records. `bynode` is reserved for future use.
"""
function profilemul(A, x; bylevel::Bool=true, bynode::Bool=false)
    d = TimingDiagnostics(; bylevel=bylevel, bynode=bynode)
    y = Vector{promote_type(eltype(x), eltype(A))}(undef, size(A, 1))
    timedmul!(y, A, x; diagnostics=d)
    return TimingReport(y, d.records, _timingmetadata(A))
end

# Access the records without exposing their storage choice.
records(r::TimingReport) = r.records

"""
    diagnostic_rows(report_or_records) -> Vector{NamedTuple}

Convert a [`TimingReport`](@ref) or vector of records to `NamedTuple` rows. Each row contains the
record fields and a derived `milliseconds` value.
"""
diagnostic_rows(r::TimingReport) = diagnostic_rows(r.records)
function diagnostic_rows(recs::AbstractVector{TimingRecord})
    return [
        (
            operation=rec.operation,
            phase=rec.phase,
            level=rec.level,
            node=rec.node,
            mode=rec.mode,
            seconds=rec.seconds,
            milliseconds=rec.seconds * 1e3,
            allocations=rec.allocations,
            bytes=rec.bytes,
            count=rec.count,
        ) for rec in recs
    ]
end
"""
    totalseconds(report_or_records, operation) -> Float64

Summed `seconds` of every record with the given `operation` (e.g. `:far`).
"""
totalseconds(r::TimingReport, operation::Symbol) = totalseconds(r.records, operation)
function totalseconds(recs::AbstractVector{TimingRecord}, operation::Symbol)
    return sum((rec.seconds for rec in recs if rec.operation === operation); init=0.0)
end
