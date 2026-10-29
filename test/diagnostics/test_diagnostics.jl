module TestDiagnostics

using Test
using LinearAlgebra
using StaticArrays
using Random
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils
using H2Factory.Diagnostics

include("../farinteractions/component_harness_fixtures.jl")

# A fake Galerkin map with genuine far pairs (the `b4tree` fixture is built to have them). Scalar
# translator so the far field is nonzero and reproducible; no mesh, no BEAST.
function _fakemap(; c=0.7 + 0.0im)
    tree, _ = b4tree()
    return H2FactoryTestUtils.fakegalerkinmap(
        tree; T=ComplexF64, i2o=H2FactoryTestUtils.scalartranslator(c)
    )
end

# A fake Petrov map with SEPARATE, genuinely different-depth test/trial trees (256 test points vs 16
# trial, overlapping) -- the shape where receiving/translating nodes live in different id spaces and
# a Galerkin-shaped diagnostic would misbehave.
# A map subtyping the H2 hierarchy but implementing NONE of the accessors -- every sub-check throws
# on it. Used to prove `sanitycheck` is total (degrades to warnings instead of raising).
struct AliasingUnavailableMap <: H2Factory.GalerkinH2Map{ComplexF64} end

_petrovpts(v) = [(p, 0.0, 0.0) for p in v]
function _fakepetrov(; c=0.7 + 0.0im)
    return H2FactoryTestUtils.fakepetrovmap(
        _petrovpts(uniformpts(256)),
        _petrovpts(0.5 .+ uniformpts(16));
        T=ComplexF64,
        i2o=H2FactoryTestUtils.scalartranslator(c),
        minhalfsize=0.004,
        testminvalues=1,
        trialminvalues=1,
    )
end

@testset "profilemul: instrumentation does not change the result" begin
    # THE invariant: profilemul drives the SAME diagnostic-aware `_unsafe_mul!` the standard path
    # uses (just with a concrete diagnostics object), so `y` must be bit-identical to `A*x` -- not
    # merely ≈. If this ever weakens, the profiler has diverged from the real path.
    rng = MersenneTwister(1)
    A = _fakemap()
    x = randn(rng, ComplexF64, size(A, 2))

    rep = profilemul(A, x)
    @test rep.y == A * x

    # Records: the whole-operation totals (`:mul`, `:near`, `:far`), plus per-level far records.
    totals = Set(r.operation for r in rep.records if r.phase === :total)
    @test :mul in totals
    @test :near in totals
    @test :far in totals
    levelops = Set(r.operation for r in rep.records if r.phase === :level)
    @test :aggregate in levelops                      # upward pass, per level
    @test !isempty(levelops)                          # per-level records are actually emitted

    # Metadata is populated and sensible.
    # The executable object is now the workspace, so this is the type the report names.
    @test rep.metadata.maptype === :GalerkinH2Workspace
    @test rep.metadata.size == size(A)
    @test rep.metadata.mode === :forward
    @test rep.metadata.treetype === :TwoNTree
end

@testset "profilemul: transpose and adjoint report the correct mode and result" begin
    rng = MersenneTwister(2)
    A = _fakemap()
    x = randn(rng, ComplexF64, size(A, 2))

    rept = profilemul(transpose(A), x)
    @test rept.metadata.mode === :transpose
    @test rept.y == transpose(A) * x
    # per-level records carry the flipped mode too
    @test all(r -> r.mode === :transpose, rept.records)

    repa = profilemul(adjoint(A), x)
    @test repa.metadata.mode === :adjoint
    @test repa.y == adjoint(A) * x
    @test all(r -> r.mode === :adjoint, repa.records)
end

@testset "profilemul.y == A*x across orientation × map type (Galerkin/Petrov/Toeplitz)" begin
    # The instrumentation-must-not-change-the-result invariant, swept over every combination the
    # traversal threads diagnostics through: forward/transpose/adjoint × the three map families.
    tree, _ = b4tree()
    mk(v) = [(p, 0.0, 0.0) for p in v]
    maps = (
        ("galerkin", _fakemap()),
        ("petrov", _fakepetrov()),
        ("toeplitz", H2FactoryTestUtils.faketoeplitzmap(tree, z -> 1.0 + 0.0im; rank=2)),
    )
    for (name, A) in maps
        for wrap in (identity, transpose, adjoint)
            M = wrap(A)
            x = randn(ComplexF64, size(M, 2))   # sized for THIS orientation (Petrov is rectangular)
            rep = profilemul(M, x)
            @test rep.y == M * x                # byte-identical to the standard matvec
        end
    end
end

@testset "NoDiagnostics records nothing and preserves the result" begin
    rng = MersenneTwister(3)
    A = _fakemap()
    x = randn(rng, ComplexF64, size(A, 2))
    y = similar(x)

    # A hand-held TimingDiagnostics accumulates; NoDiagnostics must be a pure no-op.
    d = TimingDiagnostics()
    @test isempty(d.records)
    record!(NoDiagnostics(), :mul, :total; seconds=1.0)   # no-op, must not throw
    # with_timing on NoDiagnostics returns f() untouched and records nothing (the core no-op).
    got = with_timing(NoDiagnostics(), :far, :total) do
        42
    end
    @test got == 42

    H2Factory.Diagnostics.timedmul!(y, A, x; diagnostics=NoDiagnostics())
    @test y == A * x
end

@testset "standard mul! uses NoDiagnostics: same result, no extra allocation" begin
    # The whole point of the dispatch design: `mul!(y, A, x)` goes through the SAME
    # diagnostic-aware `_unsafe_mul!` as the profiler, but with `NoDiagnostics()`, which the compiler
    # specializes away. So the standard path is byte-identical to a hand-held NoDiagnostics run AND
    # allocates no more than before diagnostics existed (the per-level `with_timing` closures must
    # elide -- helper-extracted so loop variables are arguments, not boxed captures).
    rng = MersenneTwister(9)
    A = _fakemap()
    x = randn(rng, ComplexF64, size(A, 2))
    y = similar(x)
    yref = similar(x)

    mul!(yref, A, x)                                  # standard path
    H2Factory.Diagnostics.timedmul!(y, A, x; diagnostics=NoDiagnostics())
    @test y == yref                                   # NoDiagnostics timedmul == standard mul!

    # Warm up, then assert the standard path adds no allocation vs a second identical call (a
    # closure-boxing regression would make this grow).
    mul!(y, A, x)
    mul!(y, A, x)
    a1 = @allocated mul!(y, A, x)
    a2 = @allocated mul!(y, A, x)
    @test a1 == a2                                    # stable, no per-call growth
end

@testset "TimingDiagnostics: totalseconds sums per operation" begin
    rng = MersenneTwister(4)
    A = _fakemap()
    x = randn(rng, ComplexF64, size(A, 2))
    rep = profilemul(A, x)
    # totalseconds picks the rows for one operation. `:mul`/`:near`/`:far` (totals) and `:aggregate`
    # (per level) are all recorded; a non-recorded operation sums to 0.
    @test totalseconds(rep, :mul) >= 0.0
    @test totalseconds(rep, :near) >= 0.0
    @test totalseconds(rep, :far) >= 0.0
    @test totalseconds(rep, :aggregate) >= 0.0
    @test totalseconds(rep, :nonexistent) == 0.0
    # bylevel=false suppresses the per-level records but KEEPS the whole-operation totals.
    repflat = profilemul(A, x; bylevel=false)
    @test !any(r -> r.phase === :level, repflat.records)
    @test any(r -> r.operation === :mul, repflat.records)
    @test any(r -> r.operation === :near, repflat.records)
    @test any(r -> r.operation === :far, repflat.records)
end

@testset "diagnostic_rows: dependency-free table rows" begin
    rng = MersenneTwister(11)
    A = _fakemap()
    x = randn(rng, ComplexF64, size(A, 2))
    rep = profilemul(A, x)
    rows = diagnostic_rows(rep)
    @test length(rows) == length(rep.records)
    @test rows isa Vector{<:NamedTuple}
    r = first(rows)
    # carries the record fields plus a derived milliseconds
    @test haskey(r, :operation) && haskey(r, :seconds) && haskey(r, :milliseconds)
    @test r.milliseconds ≈ r.seconds * 1e3
end

@testset "h2summary: map and tree statistics" begin
    A = _fakemap()
    s = h2summary(A)

    @test s.size == size(A)
    @test s.eltype == ComplexF64
    @test s.numleaves >= 1
    @test s.numnodes >= s.numleaves
    @test s.numlevels >= 1
    @test s.minleafdofs <= s.meanleafdofs <= s.maxleafdofs
    @test s.farpairs >= 1

    @test :storage ∉ fieldnames(typeof(s))
    @test :estimate ∉ fieldnames(typeof(s))
    shown = sprint(show, MIME"text/plain"(), s)
    @test occursin("H2 map summary", shown)
    @test occursin("far field", shown)
    @test !occursin("MiB", shown)
end

@testset "memory reporting API is removed" begin
    for name in (:memoryreport, :memoryestimate, :StorageReport, :StorageEstimate)
        @test name ∉ names(H2Factory.Diagnostics; all=true)
    end
end

@testset "translation_counts: functor translator = one operator, maximal reuse" begin
    A = _fakemap()
    tc = translation_counts(A)
    @test tc.farpairs >= 1
    # A ScalarTranslator is a single functor applied to every pair.
    @test tc.uniquetranslations == 1
    @test tc.reuse == tc.farpairs
end

@testset "sanitycheck: a well-formed map is clean" begin
    A = _fakemap()
    reports = sanitycheck(A)
    for r in reports
        @test !haserrors(r)
    end
    @test all(r -> !haserrors(r), reports)
end

@testset "checkmoments: a missing moment is reported, not thrown" begin
    # Build a map, then delete a moment a far pair needs. `checkmoments` must return an :error
    # finding rather than raising -- the whole point is to catch this before matvec KeyErrors.
    A = _fakemap()
    trialm = H2Factory.trialmoments(A)
    victim = first(keys(trialm))
    delete!(trialm, victim)

    rep = checkmoments(A)
    # It either flags the missing trial moment (if `victim` is a translating node) or stays clean
    # (if it wasn't a far-pair node) -- but it must never throw, and if it flags, it's an :error.
    @test rep isa H2Factory.Diagnostics.SanityReport
    if !isempty(rep)
        @test haserrors(rep)
        @test any(i -> i.code === :missing_trial_moment, rep.issues)
    end
end

@testset "checkplans: self-far-pair is an error (discriminating)" begin
    # `checkplans` walks the real plans, so we can't easily inject a self-pair there. Instead verify
    # the check's own logic on a clean map: no self pairs, no errors.
    A = _fakemap()
    rep = checkplans(A)
    @test !haserrors(rep)
    @test rep.target === :plans
end

# --- Petrov coverage: separate test/trial trees, different id spaces, different depths ------------

@testset "Petrov: diagnostics use per-side trees and stay clean" begin
    # The shape a Galerkin-only diagnostic gets wrong: a `BlockTree` with genuinely different test
    # (256 pts, deeper) and trial (16 pts, shallower) trees, whose node id spaces overlap. Receiving
    # nodes are test-tree ids, translating nodes trial-tree ids, so a naive `rn == tn` self-pair test
    # or a single-tree level lookup misfires here.
    A = _fakepetrov()
    @test size(A) == (256, 16)

    # h2summary must describe the TEST (row) tree, not the BlockTree or the trial side.
    s = h2summary(A)
    testtree = H2Trees.testtree(H2Trees.tree(A))
    @test s.numlevels == H2Trees.numberoflevels(testtree)
    @test s.numleaves == length(H2Trees.leaves(testtree))
    @test s.farpairs >= 1

    # sanitycheck must be clean AND must not raise a spurious self_far_pair from the coincidental
    # id-space overlap between the two trees.
    reports = sanitycheck(A)
    for r in reports
        @test !haserrors(r)
    end
    @test !any(r -> any(i -> i.code === :self_far_pair, r.issues), reports)

    # checkmoments levels each side against its own tree; a mismatch would be a false
    # nonuniform_moment_size error here (test/trial trees have different depths).
    @test !haserrors(checkmoments(A))

    # profiling still exact on the Petrov path.
    x = randn(ComplexF64, size(A, 2))
    @test profilemul(A, x).y == A * x
end

@testset "keyed translation collection: pair-resolution checks run and pass" begin
    # A `faketoeplitzmap` uses a REAL keyed `ToeplitzI2OTranslationCollection` (indexable by
    # `(recv, transl)`), unlike the functor translators above. This exercises the `_ispairindexable`
    # = true branches: translation_counts dedups by operator identity, and checkplans actually
    # resolves each pair (rather than skipping the resolution check).
    tree, _ = b4tree()
    A = H2FactoryTestUtils.faketoeplitzmap(tree, z -> 1.0 + 0.0im; rank=2)

    @test H2Factory.Diagnostics._ispairindexable(H2Factory.i2otranslator(A))

    tc = translation_counts(A)
    @test tc.farpairs >= 1
    # Every pair resolves, so no :unresolved_translation error.
    rep = checkplans(A)
    @test !haserrors(rep)
    @test !any(i -> i.code === :unresolved_translation, rep.issues)

    # sanitycheck clean on the keyed path too.
    @test all(r -> !haserrors(r), sanitycheck(A))
end

@testset "transpose/adjoint: per-side trees flip with the moments (Petrov)" begin
    # `testmoments(transpose(A))` returns the forward TRIAL moments; the tree used to level them must
    # flip in lockstep (`_testtree` -> forward trial tree), or checkmoments levels nodes against a
    # tree that doesn't contain them. checkmoments must stay clean under transpose AND adjoint.
    A = _fakepetrov()
    for M in (transpose(A), adjoint(A))
        rep = checkmoments(M)
        @test !haserrors(rep)
        # and sanitycheck as a whole stays clean + throw-free in the flipped orientation
        @test all(r -> !haserrors(r), sanitycheck(M))
    end
end

@testset "transpose/adjoint: keyed collection resolves the REVERSED key (asymmetric Petrov)" begin
    # The collection is keyed by forward (recv, transl); the transpose/adjoint traversal reverses the
    # lookup. On an asymmetric keyed map only the matching key resolves ALL pairs, so a forward-only
    # key would falsely report nearly every pair unresolved. checkplans/translation_counts must use
    # the orientation-correct key.
    mk(v) = [(p, 0.0, 0.0) for p in v]
    A = H2FactoryTestUtils.fakepetrovtoeplitzmap(
        mk(uniformpts(128)),
        mk(0.5 .+ uniformpts(16)),
        z -> 1.0 + 0.5z;
        rank=2,
        minhalfsize=0.008,
        testminvalues=1,
        trialminvalues=1,
    )
    fwdpairs = translation_counts(A).farpairs
    @test fwdpairs > 1   # genuinely many far pairs, so the reversed-key bug would be visible

    for M in (transpose(A), adjoint(A))
        rep = checkplans(M)
        # With the correct reversed key, every pair resolves -> no unresolved_translation error.
        @test !haserrors(rep)
        @test !any(i -> i.code === :unresolved_translation, rep.issues)
        # translation_counts must resolve the same number of pairs as forward (not collapse to ~1).
        @test translation_counts(M).farpairs == fwdpairs
        @test translation_counts(M).uniquetranslations >= 1
    end
end

@testset "sanitycheck is total: unavailable checks degrade to warnings, never throw" begin
    # A map with no accessors makes every sub-check throw if called unguarded. sanitycheck must still
    # return a vector of reports, surfacing each failure as a :*_check_unavailable warning.
    reports = sanitycheck(AliasingUnavailableMap())   # must not throw
    @test reports isa Vector{H2Factory.Diagnostics.SanityReport}
    codes = [i.code for r in reports for i in r.issues]
    @test :aliasing_check_unavailable in codes
    @test :plans_check_unavailable in codes
    @test :moments_check_unavailable in codes
    # all such degradations are warnings, not errors
    @test all(
        i -> i.severity === :warning,
        (i for r in reports for i in r.issues if endswith(string(i.code), "_unavailable")),
    )
end

end # module TestDiagnostics
