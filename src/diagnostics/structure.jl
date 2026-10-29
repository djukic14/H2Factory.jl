# Read-only checks of an assembled H2 map.

# Return the far pairs visited by the active traversal plan.
function _farpairs(A)
    pairs = Tuple{Int,Int}[]
    disaggplan = disaggregationplan(A)
    if disaggplan isa H2Trees.DisaggregateTranslatePlan
        for level in H2Trees.disaggregationlevels(disaggplan)
            for node in H2Trees.disaggregationnodes(disaggplan, level)
                for tn in H2Trees.translatingnodes(disaggplan, node)
                    push!(pairs, (node, tn))
                end
            end
        end
    else
        aggplan = aggregationplan(A)
        for level in H2Trees.aggregationlevels(aggplan)
            for rn in H2Trees.receivingnodes(aggplan, level)
                for tn in H2Trees.translatingnodes(aggplan, rn, level)
                    push!(pairs, (rn, tn))
                end
            end
        end
    end
    return pairs
end

# A keyed collection can be checked pair by pair; a single translator applies to every pair.
_ispairindexable(i2o) = hasmethod(Base.getindex, Tuple{typeof(i2o),Tuple{Int,Int}})

# These accessors flip with the map orientation, just like the moment accessors.
_testtree(A) = H2Trees.testtree(A)
_trialtree(A) = H2Trees.trialtree(A)

# Translation collections use forward keys. Reverse the pair for transpose and adjoint maps.
_translationkey(A, rn, tn) = _translationkey(maptrait(A), rn, tn)
_translationkey(::isForward, rn, tn) = (rn, tn)
_translationkey(::Union{isTranspose,isAdjoint}, rn, tn) = (tn, rn)

"""
    TranslationCounts

Counts far pairs and distinct translation operators. `reuse` is
`farpairs / uniquetranslations`; higher values mean more reuse.
"""
struct TranslationCounts
    farpairs::Int
    uniquetranslations::Int
    reuse::Float64
end

"""
    translation_counts(A) -> TranslationCounts

Count far pairs, distinct translation operators by identity, and their reuse.
"""
function translation_counts(A)
    pairs = _farpairs(A)
    npairs = length(pairs)
    # Work with the coupling collection whether or not the translator wraps it.
    i2o = coupling(i2otranslator(A))
    if _ispairindexable(i2o)
        uniq = Set{UInt}(objectid(i2o[_translationkey(A, rn, tn)]) for (rn, tn) in pairs)
        nuniq = length(uniq)
    else
        # A single translator is reused for every pair.
        nuniq = npairs == 0 ? 0 : 1
    end
    return TranslationCounts(npairs, nuniq, nuniq == 0 ? NaN : npairs / nuniq)
end

"""
    H2Summary

Map size, tree shape, leaf occupancy, and far-pair counts.
"""
struct H2Summary
    maptype::Symbol
    size::Tuple{Int,Int}
    eltype::DataType
    treetype::Symbol
    numlevels::Int
    numleaves::Int
    numnodes::Int
    minleafdofs::Int
    maxleafdofs::Int
    meanleafdofs::Float64
    farpairs::Int
    uniquetranslations::Int
end

"""
    h2summary(A) -> H2Summary

Return a structural summary of `A`. For a Petrov-Galerkin map, tree fields describe the
test tree.
"""
function h2summary(A)
    # Rows belong to the test side, so summarize its tree.
    tree = _testtree(A)
    leaves = H2Trees.leaves(tree)
    dofs = [length(H2Trees.values(tree, leaf)) for leaf in leaves]
    tc = translation_counts(A)
    return H2Summary(
        nameof(typeof(A)),
        size(A),
        eltype(A),
        nameof(typeof(tree)),
        H2Trees.numberoflevels(tree),
        length(leaves),
        H2Trees.numberofnodes(tree),
        isempty(dofs) ? 0 : minimum(dofs),
        isempty(dofs) ? 0 : maximum(dofs),
        isempty(dofs) ? 0.0 : sum(dofs) / length(dofs),
        tc.farpairs,
        tc.uniquetranslations,
    )
end

"""
    DiagnosticIssue(severity, code, message)

One finding from a structural check. `severity` is `:error`, `:warning`, or `:info`; `code` identifies
the condition and `message` describes it.
"""
struct DiagnosticIssue
    severity::Symbol
    code::Symbol
    message::String
end

"""
    SanityReport(target, issues)

Result of checking `target`. An empty report is clean. [`haserrors`](@ref) returns `true` when any
issue has severity `:error`.
"""
struct SanityReport
    target::Symbol
    issues::Vector{DiagnosticIssue}
end

"""
    haserrors(report::SanityReport) -> Bool

Return `true` when `report` contains an error.
"""
haserrors(r::SanityReport) = any(i -> i.severity === :error, r.issues)
Base.isempty(r::SanityReport) = isempty(r.issues)

"""
    checkplans(A) -> SanityReport

Check the far pairs and, for keyed translation collections, verify that every pair resolves.
"""
function checkplans(A)
    issues = DiagnosticIssue[]
    pairs = _farpairs(A)
    if isempty(pairs)
        push!(
            issues,
            DiagnosticIssue(
                :warning, :no_far_pairs, "map has no far pairs; the far field is empty"
            ),
        )
    end
    i2o = coupling(i2otranslator(A))
    indexable = _ispairindexable(i2o)
    # Node ids are comparable only when both sides use the same tree.
    checkselfpairs = _testtree(A) === _trialtree(A)
    nselfpairs = 0
    nunresolved = 0
    for (rn, tn) in pairs
        checkselfpairs && rn == tn && (nselfpairs += 1)
        # A single translator applies to every pair and needs no lookup check.
        if indexable
            try
                i2o[_translationkey(A, rn, tn)]
            catch
                nunresolved += 1
            end
        end
    end
    nselfpairs > 0 && push!(
        issues,
        DiagnosticIssue(
            :error,
            :self_far_pair,
            "$nselfpairs far pair(s) translate a node with itself (a node is never well-separated from itself)",
        ),
    )
    nunresolved > 0 && push!(
        issues,
        DiagnosticIssue(
            :error,
            :unresolved_translation,
            "$nunresolved far pair(s) have no translation operator in the collection",
        ),
    )
    return SanityReport(:plans, issues)
end

"""
    checkmoments(A) -> SanityReport

Check that each far-pair node has a moment and that moments on the same level have the same size.
"""
function checkmoments(A)
    issues = DiagnosticIssue[]
    # Each side must be checked against its own tree.
    testtree = _testtree(A)
    trialtree = _trialtree(A)
    trialm = trialmoments(A)
    testm = testmoments(A)

    # Receiving nodes need test moments; translating nodes need trial moments.
    missingrecv = 0
    missingtransl = 0
    for (rn, tn) in _farpairs(A)
        haskey(testm, rn) || (missingrecv += 1)
        haskey(trialm, tn) || (missingtransl += 1)
    end
    missingrecv > 0 && push!(
        issues,
        DiagnosticIssue(
            :error,
            :missing_test_moment,
            "$missingrecv receiving node(s) have no test moment (KeyError at matvec)",
        ),
    )
    missingtransl > 0 && push!(
        issues,
        DiagnosticIssue(
            :error,
            :missing_trial_moment,
            "$missingtransl translating node(s) have no trial moment (KeyError at matvec)",
        ),
    )

    # Translation operators require one moment size per level.
    for (label, moments, sidetree) in
        ((:test, testm, testtree), (:trial, trialm, trialtree))
        bylevel = Dict{Int,Any}()
        for (node, m) in moments
            lvl = H2Trees.level(sidetree, node)
            sz = size(m[1])
            if haskey(bylevel, lvl)
                if bylevel[lvl] != sz
                    push!(
                        issues,
                        DiagnosticIssue(
                            :error,
                            :nonuniform_moment_size,
                            "$label moments at level $lvl have differing sizes ($(bylevel[lvl]) vs $sz)",
                        ),
                    )
                    break
                end
            else
                bylevel[lvl] = sz
            end
        end
    end
    return SanityReport(:moments, issues)
end

# Turn an unavailable check into a warning so `sanitycheck` can return the remaining reports.
function _guardedcheck(f, A, target::Symbol, unavailablecode::Symbol)
    try
        return f(A)
    catch err
        return SanityReport(
            target,
            [
                DiagnosticIssue(
                    :warning,
                    unavailablecode,
                    "$target check could not run on this map ($(sprint(showerror, err)))",
                ),
            ],
        )
    end
end

"""
    sanitycheck(A) -> Vector{SanityReport}

Run [`checkplans`](@ref), [`checkmoments`](@ref), and [`checkmomentaliasing`](@ref). A check that
cannot run becomes a warning rather than an exception.
"""
function sanitycheck(A)
    reports = SanityReport[
        _guardedcheck(checkplans, A, :plans, :plans_check_unavailable),
        _guardedcheck(checkmoments, A, :moments, :moments_check_unavailable),
    ]
    # Convert the aliasing report and omit it when it is empty.
    aliasreport = _guardedcheck(A, :moment_aliasing, :aliasing_check_unavailable) do a
        aliasing = checkmomentaliasing(a)
        return SanityReport(
            :moment_aliasing,
            [
                DiagnosticIssue(f.severity, f.code, f.message) for
                f in aliasing.findings if f.severity !== :info
            ],
        )
    end
    isempty(aliasreport) || push!(reports, aliasreport)
    return reports
end

# Admissibility

"""
    admissibilityreport(A::AbstractH2Maps; kwargs...)

Check the translation scheme with `H2Trees.checkadmissibility`.
Forward keywords, defaulting `isnear` to the map's predicate.
Use the forward map: transpose/adjoint share its geometry.
This may visit every far pair; run it outside multiplication.
"""
function admissibilityreport(A::AbstractH2Maps; kwargs...)
    F = forwardmap(A)
    plans = (
        trialaggregationplan=trialaggregationplan(F),
        testdisaggregationplan=testdisaggregationplan(F),
        testaggregationplan=testaggregationplan(F),
        trialdisaggregationplan=trialdisaggregationplan(F),
    )
    return H2Trees.checkadmissibility(
        H2Trees.tree(F), plans; isnear=H2Trees.isnear(F), kwargs...
    )
end
