# Check pooled moment aliasing without running a matvec. Pointer extents show which arrays overlap;
# plan lifetimes determine whether each overlap is safe. Findings name the affected nodes and ranges.
"""
    MomentExtent

Describe one wrapped array in the half-open byte range `[first, last)`. `role` distinguishes
ordinary moment arrays from padded Toeplitz arrays.
"""
struct MomentExtent
    node::Int
    level::Int
    side::Symbol      # :test | :trial
    role::Symbol      # :moment | :toeplitz
    component::Int
    storage::Symbol   # :stored | :transienteven | :transientodd | :outofpool
    first::UInt
    last::UInt
end

overlaps(a::MomentExtent, b::MomentExtent) = a.first < b.last && b.first < a.last

"""
    MomentAliasingFinding

One finding, with a stable `code` and `:error`, `:warning`, or `:info` severity.
"""
struct MomentAliasingFinding
    severity::Symbol
    code::Symbol
    message::String
    extents::Vector{MomentExtent}
end

"""
    MomentAliasingReport

`ok` is true when no `:error`-severity finding was recorded.
"""
struct MomentAliasingReport
    ok::Bool
    findings::Vector{MomentAliasingFinding}
end

function MomentAliasingReport(findings::Vector{MomentAliasingFinding})
    return MomentAliasingReport(!any(f -> f.severity === :error, findings), findings)
end

function Base.show(io::IO, r::MomentAliasingReport)
    if r.ok && isempty(r.findings)
        print(io, "MomentAliasingReport: ok (no findings)")
        return nothing
    end
    print(
        io,
        "MomentAliasingReport: ",
        r.ok ? "ok" : "FAILED",
        ", $(length(r.findings)) finding(s)",
    )
    for f in r.findings
        print(io, "\n  [", f.severity, "] ", f.code, ": ", f.message)
    end
    return nothing
end

# --- pools ----------------------------------------------------------------------------------------

# Each pool is also the GC anchor for the moment views it backs.
struct _Pool
    name::Symbol
    storage::Symbol
    first::UInt
    last::UInt
end

function _pools(buffers)
    return (
        _Pool(
            :evenbuffer,
            :transienteven,
            UInt(pointer(buffers.evenbuffer)),
            UInt(pointer(buffers.evenbuffer)) +
            UInt(length(buffers.evenbuffer) * sizeof(eltype(buffers.evenbuffer))),
        ),
        _Pool(
            :oddbuffer,
            :transientodd,
            UInt(pointer(buffers.oddbuffer)),
            UInt(pointer(buffers.oddbuffer)) +
            UInt(length(buffers.oddbuffer) * sizeof(eltype(buffers.oddbuffer))),
        ),
        _Pool(
            :aggregationbuffer,
            :stored,
            UInt(pointer(buffers.aggregationbuffer)),
            UInt(pointer(buffers.aggregationbuffer)) + UInt(
                length(buffers.aggregationbuffer) *
                sizeof(eltype(buffers.aggregationbuffer)),
            ),
        ),
    )
end

# `:outofpool` also covers an extent that straddles two pools.
function _classify(pools, first::UInt, last::UInt)
    for p in pools
        first >= p.first && last <= p.last && return p.storage
    end
    return :outofpool
end

_arraybytes(a) = (UInt(pointer(a)), UInt(pointer(a)) + UInt(length(a) * sizeof(eltype(a))))

# --- extraction -----------------------------------------------------------------------------------

_extentroles(m) = ((:moment, moments(m)),)
function _extentroles(m::AbstractToeplitzMoment)
    return ((:moment, moments(m)), (:toeplitz, m.toeplitzmoments))
end

"""
    momentextents(momentsdict, tree, side, pools) -> Vector{MomentExtent}

Every wrapped array backing every moment in `momentsdict`, as byte extents classified by pool. `side`
is `:test` or `:trial`. Levels come from `tree`, so the extents carry the parity the reuse argument
turns on.
"""
function momentextents(momentsdict, tree, side::Symbol, pools)
    extents = MomentExtent[]
    for (node, m) in momentsdict
        level = H2Trees.level(tree, node)
        for (role, arrays) in _extentroles(m)
            for i in eachindex(arrays)
                first, last = _arraybytes(arrays[i])
                push!(
                    extents,
                    MomentExtent(
                        node,
                        level,
                        side,
                        role,
                        i,
                        _classify(pools, first, last),
                        first,
                        last,
                    ),
                )
            end
        end
    end
    return extents
end

# --- Tier 1: physical -----------------------------------------------------------------------------

_desc(e::MomentExtent) =
    "$(e.side) node $(e.node) (level $(e.level), $(e.role) component $(e.component), " *
    "$(e.storage), bytes [$(e.first), $(e.last)))"

"""
    requiredalignment(extent) -> Int

The byte alignment `extent` must satisfy, dispatched on what reads it.

`:toeplitz` extents are executed by an FFTW plan built on a *different* array, so their alignment class
is part of FFTW's contract and must clear the widest SIMD class ([`TOEPLITZALIGNMENT`](@ref)); a
mismatch is an `assert_applicable` throw mid-matvec. `:moment` extents are read by nothing
alignment-sensitive and only need [`MOMENTALIGNMENT`](@ref).
"""
function requiredalignment(e::MomentExtent)
    return e.role === :toeplitz ? TOEPLITZALIGNMENT : MOMENTALIGNMENT
end

"""
    checkextents(extents) -> Vector{MomentAliasingFinding}

Tier 1. Judges the memory layout on its own terms, with no knowledge of the traversal:

  - every extent lies wholly inside exactly one pool (`:out_of_pool` otherwise -- this also catches an
    extent straddling two pools);
  - no two extents at the **same level** overlap (they are written concurrently under `@tasks`);
  - a **stored** extent overlaps nothing at all (it must survive until I2O/disaggregation reads it);
  - a transient overlap is legal **only** when both extents are transient, in the same parity pool,
    and at different levels -- anything else is an overlap the even/odd scheme does not sanction.

O(n^2) in the number of extents, which is fine for a diagnostic.
"""

function checkextents(extents::Vector{MomentExtent})
    findings = MomentAliasingFinding[]

    for e in extents
        align = requiredalignment(e)
        if !iszero(e.first % align)
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :misaligned_extent,
                    "extent is not $align-byte aligned (offset $(e.first % align) past a " *
                    "boundary)" *
                    (
                        if e.role === :toeplitz
                            "; FFTW plans are built on a separate array and reject a different " *
                            "alignment class, so this throws inside `modulate_in!`"
                        else
                            ""
                        end
                    ) *
                    ": $(_desc(e))",
                    [e],
                ),
            )
        end
        if e.storage === :outofpool
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :out_of_pool,
                    "extent lies outside every pool, or straddles two: $(_desc(e))",
                    [e],
                ),
            )
        end
    end

    for i in eachindex(extents), j in (i + 1):lastindex(extents)
        a, b = extents[i], extents[j]
        overlaps(a, b) || continue

        # Same node+role+component on both sides is the same array seen twice, not an alias.
        if a.node == b.node &&
            a.side == b.side &&
            a.role == b.role &&
            a.component == b.component
            continue
        end

        pair = [a, b]
        if a.storage === :stored || b.storage === :stored
            # Side-independent, and the one class the sequential-phase argument below does NOT excuse:
            # a stored moment persists the whole matvec (it is read later by I2O or disaggregation), so
            # it spans both sides' phases and must alias nothing -- transient or stored, same side or
            # cross side.
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :stored_overlap,
                    "a stored moment shares memory -- stored moments are read later by I2O/" *
                    "disaggregation and must alias nothing: $(_desc(a)) vs $(_desc(b))",
                    pair,
                ),
            )
        elseif a.side !== b.side
            # Both transient (stored handled above), opposite sides -- benign, whatever the level or
            # parity, reported as `:info` not an error. This must be tested BEFORE the same-level and
            # parity branches: those diagnose hazards *within one side's traversal* (a concurrent
            # `@tasks` write, or same-parity-pool reuse of a still-live level), and a cross-side pair
            # is subject to neither.
            #
            # Why benign is an execution-order fact the extent geometry cannot show, so it is stated
            # rather than inferred: the MVP processes the two sides' transient phases *sequentially*.
            # The source tree's aggregation completes -- its transients born and dead -- before the
            # target-side traversal (which carries I2O) begins, and I2O reads only the *stored*
            # translating nodes across the cut (guaranteed by `checktranslatingnodesstored`), never a
            # transient. So a test transient and a trial transient are never live at the same instant;
            # sharing memory cannot corrupt, and being on different trees they are never co-written by
            # one `@tasks` loop either. Verified empirically -- on every geometry where this fires
            # (including same-level cross-side pairs), forward and transpose `farmul!` match the dense
            # reference to ~1e-16, single- and multi-threaded.
            #
            # An earlier draft made this an `:error` on a plausible-but-false lifetime argument, and
            # also let same-level cross-side pairs fall through to `:same_level_overlap`; the
            # measurement corrected both. It rests on that sequential ordering -- if a future MVP
            # interleaved the two sides' transient phases or had I2O read a transient, redo this --
            # which is why it is still surfaced (as info) rather than dropped.
            push!(
                findings,
                MomentAliasingFinding(
                    :info,
                    :cross_side_transient_overlap,
                    "test- and trial-side transient moments share memory (benign under sequential " *
                    "aggregate-then-translate ordering; see aliasing.jl): $(_desc(a)) vs $(_desc(b))",
                    pair,
                ),
            )
        elseif a.level == b.level
            # Same side, same level -- written concurrently within one `@tasks` loop, a real race.
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :same_level_overlap,
                    "two same-side extents at level $(a.level) share memory; same-level nodes are " *
                    "written concurrently under `@tasks`: $(_desc(a)) vs $(_desc(b))",
                    pair,
                ),
            )
        elseif a.storage !== b.storage
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :parity_mismatch_overlap,
                    "extents in different parity pools share memory: $(_desc(a)) vs $(_desc(b))",
                    pair,
                ),
            )
        elseif iseven(a.level) != iseven(b.level)
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :parity_mismatch_overlap,
                    "extents of opposite level parity share memory; only same-parity levels may " *
                    "reuse storage: $(_desc(a)) vs $(_desc(b))",
                    pair,
                ),
            )
        end
        # else: both transient, same side, same parity, different levels -- exactly the sanctioned
        # reuse the even/odd scheme exists to perform.
    end

    return findings
end

# --- Tier 2: lifetime -----------------------------------------------------------------------------

# Parity reuse requires contiguous processed levels.
function _checkcontiguous(levels, code::Symbol, what::String)
    findings = MomentAliasingFinding[]
    isempty(levels) && return findings
    sorted = sort(collect(levels))
    for i in 2:lastindex(sorted)
        if sorted[i] != sorted[i - 1] + 1
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    code,
                    "$what levels are not contiguous: gap between $(sorted[i - 1]) and " *
                    "$(sorted[i]). Consecutive processed levels must alternate parity, or a " *
                    "parent and its own children land in the same parity pool.",
                    MomentExtent[],
                ),
            )
        end
    end
    return findings
end

# Only deferred I2O reads require stored moments. Fused translation reads a moment at its own
# aggregation level, before the temporary pool is reused.
_deferredtranslatingnodes(::Any) = nothing

function _deferredtranslatingnodes(plan::DisaggregateTranslatePlan)
    nodes = Set{Int}()
    for level in H2Trees.disaggregationlevels(plan)
        for rn in H2Trees.disaggregationnodes(plan, level)
            for tn in H2Trees.translatingnodes(plan, rn)
                push!(nodes, tn)
            end
        end
    end
    return nodes
end

"""
    checktranslatingnodesstored(translatingplan, extents, side) -> findings

Every node whose moment is read by a *deferred* I2O translation must have moment extents, all in
stored memory.

A deferred translating node left in transient scratch is overwritten by the next same-parity level long
before the translate phase reads it -- the aggregation pass has already moved on. The result is a
silently wrong answer, not a crash, and from the outside it is indistinguishable from a translator bug.
This is the exact failure class the Petrov `mul!` investigation burned time on.

Applies only where the translation is deferred (see `_deferredtranslatingnodes`); a fused
`AggregateTranslatePlan` reads its moments in the same pass that writes them, so transient storage is
correct there and no requirement is imposed.

`side` names the side that owns the translating nodes' moments: a deferred translation is performed by
a *disaggregation* plan but reads the moments the *aggregating* side produced, so the two are opposite.
"""
function checktranslatingnodesstored(translatingplan, extents, side::Symbol)
    findings = MomentAliasingFinding[]
    translating = _deferredtranslatingnodes(translatingplan)
    translating === nothing && return findings

    extentsbynode = Dict{Int,Vector{MomentExtent}}()
    for e in extents
        e.side === side || continue
        push!(get!(() -> MomentExtent[], extentsbynode, e.node), e)
    end

    for node in translating
        nodeextents = get(extentsbynode, node, MomentExtent[])
        if isempty(nodeextents)
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :missing_translating_moment,
                    "node $node is read by deferred I2O translation but has no $side moment extents",
                    MomentExtent[],
                ),
            )
        end
        for e in nodeextents
            e.storage === :stored && continue
            push!(
                findings,
                MomentAliasingFinding(
                    :error,
                    :transient_translating_node,
                    "node $node is read by deferred I2O translation but this component is not " *
                    "stored: $(_desc(e))",
                    [e],
                ),
            )
        end
    end
    return findings
end

# --- entry points ---------------------------------------------------------------------------------

"""
    checkmomentaliasing(A) -> MomentAliasingReport

Check the pooled moment layout of a constructed H2 map `A` for aliasing that would corrupt a matvec.

It reads moments, pools, plans, and trees through the map accessors. `A` must retain its pools; see
[`momentbuffers`](@ref). The check is read-only and runs no matvec. It takes O(n^2) in the number
of moments, so use it as a diagnostic rather than inside a matvec.

```julia
report = checkmomentaliasing(A)
report.ok || foreach(println, report.findings)
```

`report.ok` is false only for `:error`-severity findings. See [`MomentAliasingReport`](@ref).
"""
function checkmomentaliasing(A::AbstractH2Maps)
    return checkmomentaliasing(
        testmoments(A),
        trialmoments(A),
        momentbuffers(A),
        (;
            trialaggregationplan=trialaggregationplan(A),
            testdisaggregationplan=testdisaggregationplan(A),
            testaggregationplan=testaggregationplan(A),
            trialdisaggregationplan=trialdisaggregationplan(A),
        );
        trees=(test=H2Trees.testtree(A), trial=H2Trees.trialtree(A)),
    )
end

"""
    checkmomentaliasing(testmoments, trialmoments, buffers, plans; trees) -> MomentAliasingReport

Lower-level form: check a moment layout directly, without a map. `buffers` is the
`(evenbuffer, oddbuffer, aggregationbuffer)` NamedTuple the moments were wrapped out of; `plans` is
any object exposing the four plan fields (`trialaggregationplan`/`testdisaggregationplan`/
`testaggregationplan`/`trialdisaggregationplan`), e.g. an `H2Trees.PlanSet` or a `NamedTuple`; `trees`
is `(test=..., trial=...)`.

Runs Tier 1 (physical extents) then Tier 2 (traversal lifetime). See the file header for why both are
needed: geometry alone cannot tell a sanctioned overlap from a fatal one.
"""
function checkmomentaliasing(testmoments, trialmoments, buffers, plans; trees)
    pools = _pools(buffers)
    extents = vcat(
        momentextents(testmoments, trees.test, :test, pools),
        momentextents(trialmoments, trees.trial, :trial, pools),
    )

    findings = checkextents(extents)

    append!(
        findings,
        _checkcontiguous(
            H2Trees.aggregationlevels(plans.trialaggregationplan),
            :noncontiguous_aggregation_levels,
            "trial aggregation",
        ),
    )
    append!(
        findings,
        _checkcontiguous(
            H2Trees.aggregationlevels(plans.testaggregationplan),
            :noncontiguous_aggregation_levels,
            "test aggregation",
        ),
    )
    append!(
        findings,
        _checkcontiguous(
            H2Trees.disaggregationlevels(plans.testdisaggregationplan),
            :noncontiguous_disaggregation_levels,
            "test disaggregation",
        ),
    )
    append!(
        findings,
        _checkcontiguous(
            H2Trees.disaggregationlevels(plans.trialdisaggregationplan),
            :noncontiguous_disaggregation_levels,
            "trial disaggregation",
        ),
    )

    # Deferred translations live on the disaggregation plans, and they read the moments the *opposite*
    # (aggregating) side produced: forward, test-side disaggregation translates from trial moments;
    # transposed, trial-side disaggregation translates from test moments. The aggregation plans impose
    # no requirement -- either they are fused (`AggregateTranslatePlan`, translating in the same pass)
    # or they carry no translation data at all.
    append!(
        findings, checktranslatingnodesstored(plans.testdisaggregationplan, extents, :trial)
    )
    append!(
        findings, checktranslatingnodesstored(plans.trialdisaggregationplan, extents, :test)
    )

    return MomentAliasingReport(findings)
end
