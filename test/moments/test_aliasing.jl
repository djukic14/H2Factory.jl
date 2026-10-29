module TestMomentAliasing

using Test
using Random
using StaticArrays
using FFTW
using H2Factory
using H2Trees
using H2FactoryTestUtils

# Tests for `checkmomentaliasing` -- the guard on `allocatemoments.jl`'s pooled moment aliasing.
#
# The negative tests are the point of this file, not an afterthought. A checker validated only against
# good layouts proves nothing: one that never fires passes that suite perfectly. Every negative test
# below therefore asserts a **specific finding code**, so a check that silently stops working fails
# here rather than going quiet. (This is not hypothetical -- the first draft of this checker fired on
# every translating node of a provably-correct layout, and only a known-good comparison caught it.)
#
# `checkextents` is a pure function of the extents, so corrupted layouts are built directly as
# `MomentExtent` vectors -- no `unsafe_wrap` games, no real memory to corrupt, and each failure mode is
# expressible in two lines.

# Addresses are 64-aligned so they satisfy the alignment check on their own; each test then
# isolates the one property it is about.
mkext(node, level, side, storage, first, last; role=:moment, component=1) =
    H2Factory.MomentExtent(
        node, level, side, role, component, storage, UInt(first), UInt(last)
    )

codes(findings) = [f.code for f in findings]
haserror(findings, code) = any(f -> f.code === code && f.severity === :error, findings)

@testset "Tier 1 negative: each failure mode reports its own code" begin
    # Two nodes at the SAME level sharing memory: they are written concurrently under `@tasks`, so
    # this is a race regardless of parity or pool.
    f = H2Factory.checkextents([
        mkext(1, 4, :trial, :transienteven, 128, 256),
        mkext(2, 4, :trial, :transienteven, 192, 320),
    ])
    @test haserror(f, :same_level_overlap)

    # A stored moment sharing memory with anything: stored moments must survive until I2O or
    # disaggregation reads them, so they may alias nothing at all.
    f = H2Factory.checkextents([
        mkext(1, 4, :trial, :stored, 128, 256), mkext(2, 5, :trial, :transientodd, 192, 320)
    ])
    @test haserror(f, :stored_overlap)

    # Two stored moments overlapping -- same code, and the more likely real bug (a pointer that
    # failed to advance).
    f = H2Factory.checkextents([
        mkext(1, 4, :trial, :stored, 128, 256), mkext(2, 6, :trial, :stored, 128, 256)
    ])
    @test haserror(f, :stored_overlap)

    # An extent outside every pool, or straddling two: the pointer arithmetic left its region.
    f = H2Factory.checkextents([mkext(1, 4, :trial, :outofpool, 128, 256)])
    @test haserror(f, :out_of_pool)

    # Opposite level parity sharing storage: only same-parity levels may reuse, since adjacent levels
    # are live simultaneously (a parent is written while its children are read).
    f = H2Factory.checkextents([
        mkext(1, 4, :trial, :transienteven, 128, 256),
        mkext(2, 5, :trial, :transienteven, 192, 320),
    ])
    @test haserror(f, :parity_mismatch_overlap)

    # Different pools overlapping at all is incoherent -- defensive, but it is what a pool-sizing bug
    # would look like.
    f = H2Factory.checkextents([
        mkext(1, 4, :trial, :transienteven, 128, 256),
        mkext(2, 6, :trial, :transientodd, 192, 320),
    ])
    @test haserror(f, :parity_mismatch_overlap)
end

@testset "Tier 1 positive: the sanctioned reuse is not flagged" begin
    # THE case the even/odd scheme exists to perform: two transient moments, same side, same parity
    # pool, different levels, sharing memory. Levels 2 and 4 are never live at once (a transient at
    # level L dies once level L-1 has read it), so this must produce no finding at all. If this ever
    # errors, the checker has become useless -- it would reject every real layout.
    f = H2Factory.checkextents([
        mkext(1, 2, :trial, :transienteven, 128, 256),
        mkext(2, 4, :trial, :transienteven, 128, 256),
    ])
    @test isempty(f)

    # The same array seen twice is not an alias.
    e = mkext(1, 2, :trial, :transienteven, 128, 256)
    @test isempty(H2Factory.checkextents([e, e]))

    # Non-overlapping extents, whatever their class.
    f = H2Factory.checkextents([
        mkext(1, 4, :trial, :stored, 128, 256), mkext(2, 4, :trial, :stored, 256, 384)
    ])
    @test isempty(f)
end

@testset "Tier 1: cross-side transient overlap is reported but benign (info, not error)" begin
    # Same parity, different levels, opposite sides. This overlap is real (an asymmetric Petrov
    # geometry produces it), but it does not corrupt: the MVP runs the two sides' transient phases
    # sequentially -- the source tree's aggregation transients are dead before the target-side I2O
    # traversal begins, and I2O reads only the *stored* translating nodes across the cut, never these
    # transients. Verified empirically -- forward and transpose `farmul!` match the dense reference to
    # ~1e-16 on geometries where this fires. So it is surfaced as `:info` (visible for the structural
    # refactor question), not an error that fails a report. An earlier draft wrongly made it an error
    # on a plausible-but-false lifetime argument; the measurement corrected it.
    f = H2Factory.checkextents([
        mkext(1, 2, :test, :transienteven, 128, 256),
        mkext(2, 4, :trial, :transienteven, 128, 256),
    ])
    @test any(x -> x.code === :cross_side_transient_overlap, f)
    @test !haserror(f, :cross_side_transient_overlap)   # info severity, not error
    @test H2Factory.MomentAliasingReport(f).ok          # an info finding alone must not fail a report
end

@testset "Tier 2 negative: non-contiguous levels" begin
    # A gap makes two *processed* levels share a parity, co-locating a parent with its own children.
    f = H2Factory._checkcontiguous(
        [2, 3, 5], :noncontiguous_aggregation_levels, "trial aggregation"
    )
    @test haserror(f, :noncontiguous_aggregation_levels)
    @test isempty(
        H2Factory._checkcontiguous([2, 3, 4], :noncontiguous_aggregation_levels, "x")
    )
    @test isempty(H2Factory._checkcontiguous(Int[], :noncontiguous_aggregation_levels, "x"))
end

@testset "alignoffset rounds up to the next boundary" begin
    # The previous implementation added `offset % alignment` -- the distance from the *previous*
    # boundary, not to the next one. It only landed on a boundary when the remainder was exactly
    # alignment/2, and was masked in practice because 16-byte eltypes off a 16-aligned pool give a
    # remainder of 0 (the branch never fired) and 8-byte eltypes give 0 or 8 (accidentally correct).
    # A 4-byte eltype would have misaligned. These cases pin the arithmetic directly.
    for r in 0:15
        @test H2Factory.alignoffset(r, 16) == (16 - r) % 16
        @test iszero((r + H2Factory.alignoffset(r, 16)) % 16)   # actually aligned afterwards
    end
    @test H2Factory.alignoffset(0, 16) == 0        # already aligned -> no-op
    @test H2Factory.alignoffset(4, 16) == 12       # the case the old formula got wrong (gave 4)
    @test H2Factory.alignoffset(8, 16) == 8        # the case it got right by luck
    for a in (4, 8, 16), r in 0:(2a)
        @test iszero((r + H2Factory.alignoffset(r, a)) % a)
    end
end

@testset "wrapbuffer produces aligned, in-bounds, non-overlapping arrays" begin
    # Exercises the real wrap path per eltype width, including the 4-byte case the old alignment
    # arithmetic would have misaligned.
    for T in (Float32, Float64, ComplexF64)
        buf = Vector{T}(undef, 4096)
        p0 = pointer(buf)
        p, m = H2Factory.wrapbuffer(p0, p0, H2Factory.Moment1D{T}, (4,))
        a = H2Factory.moments(m)[1]
        @test iszero(UInt(pointer(a)) % 16)                      # aligned
        @test UInt(pointer(a)) >= UInt(p0)                       # in bounds
        @test UInt(p) <= UInt(p0) + UInt(length(buf) * sizeof(T))
        @test length(a) == 4
    end
end

@testset "alignmentslack is sufficient: wrapping never exceeds the budget" begin
    # THE invariant that keeps the pools safe. `calculatememoryrequirements` sizes each pool as
    # `numberofentries + alignmentslack` per moment; if `wrapbuffer` can ever consume more padding
    # than `alignmentslack` budgets, the last moment of a pool walks off the end -- silent memory
    # corruption, strictly worse than the FFTW throw the wider alignment prevents.
    #
    # Swept over eltype widths because the budget is counted in *elements* while the padding is in
    # *bytes*: the count must scale with 1/sizeof(T). ComplexF32 (8 bytes) is the case a fixed element
    # count under-budgets, and nothing else in the suite exercises it -- iFMM always uses ComplexF64.
    for T in (ComplexF64, ComplexF32, Float64, Float32)
        for (MomentT, msize) in (
            (H2Factory.Moment1D{T}, (4,)),
            (H2Factory.Moment2D{T}, (4,)),
            (H2Factory.Moment3D{T}, (5,)),
        )
            budget =
                H2Factory.numberofentries(MomentT(msize)) +
                H2Factory.alignmentslack(MomentT(msize))
            nmoments = 8
            pool = Vector{T}(undef, budget * nmoments + 64)
            GC.@preserve pool begin
                p = pointer(pool)
                limit = UInt(p) + UInt(budget * nmoments * sizeof(T))
                for _ in 1:nmoments
                    p, _m = H2Factory.wrapbuffer(p, p, MomentT, msize)
                end
                # Every moment fit inside its own share of the budget.
                @test UInt(p) <= limit
            end
        end
    end
end

@testset "alignmentslack: Toeplitz budget covers the wider alignment, per eltype" begin
    # A Toeplitz moment wraps two arrays per component and aligns the toeplitz one to 64, so it needs
    # a strictly bigger budget than a plain moment of the same shape. Pin both the ordering and the
    # sufficiency, per eltype width.
    for T in (ComplexF64, ComplexF32)
        plain = H2Factory.Moment1D{T}((4,))
        toep = H2Factory.ToeplitzMoment1D{T}(((4,), (7,)))
        @test H2Factory.alignmentslack(toep) > H2Factory.alignmentslack(plain)
        # Worst case padding actually needed, in elements: one 16-boundary + one 64-boundary per
        # component. The budget must cover it.
        needed =
            cld(H2Factory.MOMENTALIGNMENT - 1, sizeof(T)) +
            cld(H2Factory.TOEPLITZALIGNMENT - 1, sizeof(T))
        @test H2Factory.alignmentslack(toep) >= needed * length(eachindex(toep))
    end
end

@testset "wrapped Toeplitz arrays are FFTW-plan-compatible" begin
    # FFTW is alignment-sensitive in a way that bites *here specifically*: `modulate_in!` applies
    # `fftplan * toeplitzmoment(in, i)` -- the plan is built in `ext/H2Toeplitz` on a freshly allocated
    # array, but executed against an `unsafe_wrap`ped view into the moment pool. FFTW.jl's
    # `assert_applicable` compares `alignment_of(X)` to the plan's recorded alignment and throws
    # "FFTW plan applied to array with wrong memory alignment" when they differ. So the pool's wrap
    # offsets are load-bearing for FFTW, not merely a performance detail, and any change to the
    # alignment arithmetic must preserve the plan-compatibility class.
    #
    # This pins that: a plan built on a standalone buffer must execute against every wrapped array the
    # allocator hands out, at each of several pool offsets.
    T = ComplexF64
    tsize = (7, 1, 1)
    plan = FFTW.plan_fft!(Array{T,3}(undef, tsize))

    pool = Vector{T}(undef, 8192)
    GC.@preserve pool begin
        p = pointer(pool)
        for _ in 1:8
            p = H2Factory._align(p)
            wrapped = unsafe_wrap(Array{T,3}, p, tsize)
            # The load-bearing assertion: FFTW accepts this array for a plan built elsewhere. If the
            # wrap offset ever produced an incompatible alignment class, `assert_applicable` throws
            # here rather than silently computing a wrong transform.
            wrapped .= one(T)
            @test (plan * wrapped) === wrapped
            @test FFTW.alignment_of(wrapped) == plan.ialign
            p += prod(tsize) * sizeof(T)
        end
    end
end

@testset "report ok reflects error-severity findings only" begin
    @test H2Factory.MomentAliasingReport(H2Factory.MomentAliasingFinding[]).ok
    err = H2Factory.MomentAliasingFinding(:error, :some_code, "m", H2Factory.MomentExtent[])
    @test !H2Factory.MomentAliasingReport([err]).ok
    info = H2Factory.MomentAliasingFinding(:info, :some_code, "m", H2Factory.MomentExtent[])
    @test H2Factory.MomentAliasingReport([info]).ok
end

# --- positive: real pooled layouts, straight from the allocator iFMM uses --------------------------

uniformpts(N) = [SVector((i + 0.5) / N, 0.0, 0.0) for i in 0:(N - 1)]

function reallayout(tree, mode)
    plans = H2Trees.buildplans(tree; builder=H2Trees.PlanBuilder(; aggregationmode=mode))
    momentsize(tr, level) = (4,)
    MomentType = level -> H2Factory.Moment1D{ComplexF64}
    buffers, testm, trialm = H2Factory.storagemomentsplan(
        plans.testaggregationplan, plans.trialaggregationplan, tree, MomentType, momentsize
    )
    return buffers, testm, trialm, plans
end

@testset "deferred translation checks every required moment extent" begin
    tree = H2Trees.buildtree(
        uniformpts(128);
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    plans = H2Trees.buildplans(
        tree; builder=H2Trees.PlanBuilder(; aggregationmode=H2Trees.AggregateMode())
    )
    plan = plans.testdisaggregationplan
    translating = H2Factory._deferredtranslatingnodes(plan)
    @test !isempty(translating)
    node = first(translating)
    level = H2Trees.level(tree, node)

    missing = H2Factory.checktranslatingnodesstored(plan, H2Factory.MomentExtent[], :trial)
    @test haserror(missing, :missing_translating_moment)
    @test !H2Factory.MomentAliasingReport(missing).ok

    partial = [
        mkext(node, level, :trial, :transienteven, 128, 192; component=1),
        mkext(node, level, :trial, :stored, 256, 320; component=2),
    ]
    findings = H2Factory.checktranslatingnodesstored(plan, partial, :trial)
    @test haserror(findings, :transient_translating_node)
    @test any(
        f -> f.code === :transient_translating_node && f.extents[1].component == 1, findings
    )
end

@testset "real Galerkin layouts are clean (both aggregation modes)" begin
    tree = H2Trees.buildtree(
        uniformpts(256);
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        buffers, testm, trialm, plans = reallayout(tree, mode)
        report = H2Factory.checkmomentaliasing(
            testm, trialm, buffers, plans; trees=(test=tree, trial=tree)
        )
        @test report.ok
        # Every extent must land in a pool -- an `:out_of_pool` here would mean the wrap arithmetic
        # escaped its buffer, which is a live memory-corruption bug rather than a policy question.
        @test !haserror(report.findings, :out_of_pool)
        @test !haserror(report.findings, :stored_overlap)
        @test !haserror(report.findings, :same_level_overlap)
    end
end

@testset "real Toeplitz layout: alignment and pool budget hold through the allocator" begin
    # The case the split alignment exists for. A Toeplitz moment wraps *two* arrays per component and
    # aligns the toeplitz one to FFTW's wider class, so this exercises both the 64-byte alignment and
    # the `alignmentslack` budget that had to grow to cover it -- widening the alignment without
    # widening the budget would overrun the pool, which is worse than the throw it prevents.
    tree = H2Trees.buildtree(
        uniformpts(256);
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    plans = H2Trees.buildplans(tree)

    rank = 4
    momentsize(tr, level) = ((rank,), (2rank - 1,))   # (moment size, toeplitz size)
    MomentType = level -> H2Factory.ToeplitzMoment1D{ComplexF64}

    buffers, testm, trialm = H2Factory.storagemomentsplan(
        plans.testaggregationplan, plans.trialaggregationplan, tree, MomentType, momentsize
    )
    report = H2Factory.checkmomentaliasing(
        testm, trialm, buffers, plans; trees=(test=tree, trial=tree)
    )
    @test report.ok
    # Specifically: nothing misaligned, and nothing escaped its pool (the budget covered the padding).
    @test !haserror(report.findings, :misaligned_extent)
    @test !haserror(report.findings, :out_of_pool)

    # The toeplitz arrays really are present and really are on the wider boundary -- otherwise this
    # testset would pass by simply not extracting them.
    pools = H2Factory._pools(buffers)
    extents = H2Factory.momentextents(trialm, tree, :trial, pools)
    toeplitz = filter(e -> e.role === :toeplitz, extents)
    @test !isempty(toeplitz)
    @test all(e -> iszero(e.first % H2Factory.TOEPLITZALIGNMENT), toeplitz)
end

@testset "real layout: extents are classified, not silently empty" begin
    # Guards against the checker passing because it extracted nothing at all.
    tree = H2Trees.buildtree(
        uniformpts(256);
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    buffers, testm, trialm, plans = reallayout(tree, H2Trees.AggregateMode())
    pools = H2Factory._pools(buffers)
    extents = vcat(
        H2Factory.momentextents(testm, tree, :test, pools),
        H2Factory.momentextents(trialm, tree, :trial, pools),
    )
    @test length(extents) == length(testm) + length(trialm)
    @test all(e -> e.storage !== :outofpool, extents)
    # Both storage classes are genuinely exercised: the `AggregatePlan` side stores, the fused
    # `AggregateTranslatePlan` side is entirely transient.
    @test any(e -> e.storage === :stored, extents)
    @test any(e -> e.storage in (:transienteven, :transientodd), extents)
end

# Number of aggregation nodes per level whose moment is NOT stored, i.e. lives in a shared even/odd
# pool. An `AggregateTranslatePlan` stores nothing, so all of its nodes are transient.
function _transientsperlevel(plan)
    if plan isa H2Trees.AggregateTranslatePlan
        return Dict(
            level => length(H2Trees.aggregationnodes(plan, level)) for
            level in H2Trees.aggregationlevels(plan)
        )
    end
    return Dict(
        level => count(
            node -> !H2Trees.storenode(plan, node),
            H2Trees.aggregationnodes(plan, level),
        ) for level in H2Trees.aggregationlevels(plan)
    )
end

@testset "cross-side transient level clashes occur and are benign (the Phase 3 answer)" begin
    # `_storagemomentsplan` hands both aggregation plans the same `buffers`, sizes the transient pools
    # `max(a, at)` not `a + at`, and `materializemoments` restarts at the pool base per level for each
    # plan -- so test and trial transients are laid down from the same address, and their levels are NOT
    # guaranteed disjoint. An earlier draft assumed they were and treated a clash as corruption; this
    # test encodes the measured answer instead: clashes happen, and they do not corrupt, because the
    # MVP runs the two sides' transient phases sequentially (the source aggregation's transients are
    # dead before the target-side I2O begins, and I2O reads only stored translating nodes across the
    # cut). The proof is farmul! itself: on a clashing layout it still matches the dense reference.
    maxerr(y, yh) = maximum(abs.(y .- yh)) / max(maximum(abs.(y)), eps())

    clashes = 0
    aggregateplanwithtransients = false
    # Asymmetric many-test/few-trial at matched minvalues is the regime where the AggregatePlan side
    # carries transients; *varying* the counts is what makes cross-side level clashes actually appear
    # (fixed sizes rarely clash). Seeded so the search is deterministic -- this exact seed yields
    # several clashing layouts, verified.
    Random.seed!(0)
    for _ in 1:24
        testpts = [SVector{3,Float64}(randn(), randn(), randn()) for _ in 1:rand(64:256)]
        trialpts = [
            SVector{3,Float64}(0.5 + randn(), randn(), randn()) for _ in 1:rand(8:64)
        ]
        for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
            # A random draw can land with EVERY test/trial pair near -- entirely legitimate for
            # these heavily overlapping clouds under the box-gap admissibility margin, which
            # requires a full smaller-half-size of clearance before a pair counts as far (see
            # `H2Trees.DEFAULTNEARGAPBOXES`). H2Trees then refuses to build an empty
            # `AggregatePlan`. Such a layout has no far field to probe at all, so skip it rather
            # than let a geometry accident abort the search; the `clashes > 0` assertion below
            # still proves enough usable layouts were found.
            A = try
                H2FactoryTestUtils.fakepetrovmap(
                    testpts,
                    trialpts;
                    rank=2,
                    testminvalues=10,
                    trialminvalues=10,
                    aggregationmode=mode,
                )
            catch err
                # H2Trees refuses an empty AggregatePlan via `throw(ArgumentError(...))` (was `error(...)`
                # -> ErrorException before the builder refactor); accept both so this stays robust.
                err isa Union{ErrorException,ArgumentError} &&
                    occursin("Empty AggregatePlan", err.msg) &&
                    continue
                rethrow()
            end
            plans = (
                testaggregationplan=H2Factory.testaggregationplan(A),
                trialaggregationplan=H2Factory.trialaggregationplan(A),
            )
            te = _transientsperlevel(plans.testaggregationplan)
            tr = _transientsperlevel(plans.trialaggregationplan)
            if plans.testaggregationplan isa H2Trees.AggregatePlan &&
                sum(values(te); init=0) > 0
                aggregateplanwithtransients = true
            elseif plans.trialaggregationplan isa H2Trees.AggregatePlan &&
                sum(values(tr); init=0) > 0
                aggregateplanwithtransients = true
            end
            if any(l -> te[l] > 0 && tr[l] > 0, intersect(keys(te), keys(tr)))
                clashes += 1
                # A clashing layout must still produce a correct matvec -- the benign claim, proven
                # not asserted. The checker flags the overlap only as :info, so the report stays ok.
                report = checkmomentaliasing(A)
                @test any(f -> f.code === :cross_side_transient_overlap, report.findings)
                @test report.ok
                F = H2FactoryTestUtils.denseh2matrix(A)
                x = randn(ComplexF64, size(F, 2))
                yh = zeros(ComplexF64, size(F, 1))
                H2Factory.farmul!(yh, A, x)
                @test maxerr(F * x, yh) < 1e-9
            end
        end
    end

    # Both premises the corrected understanding rests on, pinned so they cannot silently lapse: an
    # `AggregatePlan` really does carry transients, and cross-side level clashes really do occur (if
    # this drops to zero, the benign-ness above is no longer being exercised and should be revisited).
    @test aggregateplanwithtransients
    @test clashes > 0
end

@testset "map-level checkmomentaliasing(A) reaches the layout through the accessors" begin
    # The convenience entry point exists so this diagnostic is actually runnable on a constructed map:
    # the lower-level form needs the moments, pools, plans and trees threaded by hand, which is enough
    # friction that nobody would. It must therefore work off `A` alone, and keep working through
    # `transpose`/`adjoint` -- `momentbuffers` resolves via `forwardmap`/`maptrait`, so an override at
    # the wrong dispatch level would be silently skipped for exactly those two and fall back to a
    # `FieldError` (the trap that has already bitten the scheduler accessors once).
    tree = H2Trees.buildtree(
        uniformpts(128);
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    A = H2FactoryTestUtils.fakegalerkinmap(tree; rank=2)

    for M in (A, transpose(A), adjoint(A))
        report = checkmomentaliasing(M)          # exported: reached unqualified
        @test report isa MomentAliasingReport
        @test report.ok
    end

    # `report.ok` on a real map is only meaningful if the checker can still fail. Without this, an
    # accessor returning something empty would produce a green `ok` and look identical to success.
    @test !isempty(
        H2Factory.momentextents(
            H2Factory.trialmoments(A),
            H2Trees.trialtree(A),
            :trial,
            H2Factory._pools(H2Factory.momentbuffers(A)),
        ),
    )
end

@testset "map-level checkmomentaliasing(A) reports ok on a real Petrov (BlockTree) layout" begin
    # The Petrov path is where cross-side sharing actually happens: test and trial moments live in
    # separate pools over separate trees, and the `isBlockTree` allocator drives both. The checker
    # runs off `A` alone and must report `ok` -- through every orientation, so a `momentbuffers`
    # override at the wrong dispatch level (skipped for transpose/adjoint) is caught. `report.ok` holds
    # even when a cross_side_transient_overlap is present, because that finding is `:info`, not an
    # error (its benign-ness is proven separately in the clash testset above); the previous-level
    # errors that WOULD fail a report -- stored/same-level/parity overlaps -- must be absent.
    Random.seed!(20260716)
    testpts = [SVector{3,Float64}(randn(), randn(), randn()) for _ in 1:256]
    trialpts = [SVector{3,Float64}(0.5 + randn(), randn(), randn()) for _ in 1:16]
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        A = H2FactoryTestUtils.fakepetrovmap(
            testpts,
            trialpts;
            rank=2,
            testminvalues=10,
            trialminvalues=10,
            aggregationmode=mode,
        )
        for M in (A, transpose(A), adjoint(A))
            report = checkmomentaliasing(M)
            @test report.ok
            for code in (
                :stored_overlap, :same_level_overlap, :parity_mismatch_overlap, :out_of_pool
            )
                @test !haserror(report.findings, code)
            end
        end
    end
end

@testset "planmomentstorage describes the layout by (level, node) order and pool" begin
    # The "plan" stage of the layout/allocation/materialization split: a pure, memory-free description
    # (`MomentSlot`s) that `materializemoments` then wraps. Its order and pool assignment must match
    # what the wrap does -- the end-to-end harness proves the *values* agree; this pins the *structure*
    # directly, so a future change to either stage that desyncs them fails here.
    tree = H2Trees.buildtree(
        uniformpts(256);
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        plans = H2Trees.buildplans(
            tree; builder=H2Trees.PlanBuilder(; aggregationmode=mode)
        )
        for plan in (plans.trialaggregationplan, plans.testaggregationplan)
            slots = H2Factory.planmomentstorage(plan)

            # Exactly one slot per aggregation node, ordered by (level, node) as materialize walks it.
            expected = [
                (node, level) for level in H2Trees.aggregationlevels(plan) for
                node in H2Trees.aggregationnodes(plan, level)
            ]
            @test [(s.node, s.level) for s in slots] == expected

            # Pool assignment mirrors `_momentpool`: stored -> :aggregation, else the level's parity.
            storenode = if plan isa H2Trees.AggregatePlan
                H2Trees.StoreNodeFunctor(plan)
            else
                H2Trees.StoreNoNodeFunctor(plan)
            end
            for s in slots
                want = storenode(s.node) ? :aggregation : (iseven(s.level) ? :even : :odd)
                @test s.pool === want
            end
            # An AggregateTranslatePlan stores nothing, so it can never use the aggregation pool.
            if plan isa H2Trees.AggregateTranslatePlan
                @test all(s -> s.pool !== :aggregation, slots)
            end
        end
    end
end

@testset "materializemoments / allocatemomentbuffers reject misuse loudly" begin
    # The stages are reusable internal building blocks, so they guard their preconditions instead of
    # trusting the caller. Build real pools once, then feed hand-crafted bad inputs.
    tree = H2Trees.buildtree(
        uniformpts(128);
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.03, minvalues=0, root=1),
    )
    plans = H2Trees.buildplans(tree)
    aggplan = plans.trialaggregationplan          # AggregatePlan in AggregateMode
    atplan = plans.testaggregationplan            # AggregateTranslatePlan
    MT = level -> H2Factory.Moment1D{ComplexF64}
    msize = (tr, level) -> (2,)

    buffers = H2Factory.allocatemomentbuffers(
        aggplan, atplan, tree, tree, MT, MT, msize, msize
    )

    # Finding 2: an unknown pool symbol must error, not silently fall through to :odd.
    @test_throws ErrorException H2Factory.materializemoments(
        [H2Factory.MomentSlot(1, 2, :bogus)], tree, MT, msize, buffers
    )

    # Finding 3: slots that revisit a level (not grouped) must error -- the per-level reset would
    # otherwise corrupt that level's already-wrapped moments.
    ungrouped = [
        H2Factory.MomentSlot(10, 2, :even),
        H2Factory.MomentSlot(11, 3, :odd),
        H2Factory.MomentSlot(12, 2, :even),   # level 2 recurs
    ]
    @test_throws ErrorException H2Factory.materializemoments(
        ungrouped, tree, MT, msize, buffers
    )

    # A correctly grouped list of the same pools does NOT error (guards the guard against false trips).
    grouped = [
        H2Factory.MomentSlot(10, 2, :even),
        H2Factory.MomentSlot(12, 2, :even),
        H2Factory.MomentSlot(11, 3, :odd),
    ]
    @test H2Factory.materializemoments(grouped, tree, MT, msize, buffers) isa Dict

    # Finding 4: the two plans' moments share one typed pool, so mismatched element types must error
    # at allocation rather than produce a pool invalid for one side.
    @test_throws AssertionError H2Factory.allocatemomentbuffers(
        aggplan,
        atplan,
        tree,
        tree,
        level -> H2Factory.Moment1D{ComplexF64},
        level -> H2Factory.Moment1D{Float64},
        msize,
        msize,
    )
end

@testset "translate-side pool is sized against the translate tree, not the aggregate tree" begin
    # Pins finding 1's bug class directly. The other tests use `tree, tree` and a size functor that
    # ignores its tree argument, so they cannot tell which tree sized which pool. Here a single
    # tree-DEPENDENT functor gives the test (translate) tree BIG moments and the trial (aggregate) tree
    # SMALL ones, over a BlockTree where those trees genuinely differ. `materializemoments` wraps the
    # test side with the test tree (big), so if `allocatemomentbuffers` sized the translate pool
    # against the aggregate tree (small) -- the pre-fix behavior -- the big test moments would spill
    # past the pool and `checkmomentaliasing` would flag `:out_of_pool`. It does not: the pool is sized
    # against the translate tree, so they fit.
    testpts = [SVector{3,Float64}(randn(), randn(), randn()) for _ in 1:96]
    trialpts = [SVector{3,Float64}(0.6 + randn(), randn(), randn()) for _ in 1:96]
    allx = vcat(getindex.(testpts, 1), getindex.(trialpts, 1))
    hs = (maximum(allx) - minimum(allx)) / 8
    blocktree = H2Trees.buildtree(
        testpts,
        trialpts;
        builder=H2Trees.BlockTreeBuilder(;
            test=H2Trees.TwoNTreeBuilder(; minhalfsize=hs, minvalues=5),
            trial=H2Trees.TwoNTreeBuilder(; minhalfsize=hs, minvalues=5),
        ),
    )
    ttree = H2Trees.testtree(blocktree)
    rtree = H2Trees.trialtree(blocktree)

    plans = H2Trees.buildplans(blocktree)

    bigsize, smallsize = (10,), (2,)
    # One functor, tree-dependent: the whole point is that the size changes with the tree, so passing
    # the wrong tree to sizing produces the wrong pool.
    msize = (tr, level) -> (tr === ttree ? bigsize : smallsize)
    MT = level -> H2Factory.Moment1D{ComplexF64}

    buffers, testm, trialm = H2Factory.storagemomentsplan(
        plans.testaggregationplan,
        plans.trialaggregationplan,
        blocktree,
        MT,
        MT,
        msize,
        msize,
    )

    # materialize used each side's own tree: test moments big, trial moments small.
    @test size(H2Factory.moments(first(values(testm)))[1]) == bigsize
    @test size(H2Factory.moments(first(values(trialm)))[1]) == smallsize

    # and the pools hold them in bounds -- the discriminating assertion. A translate pool sized against
    # the aggregate (small) tree would leave these big test moments out of pool.
    report = H2Factory.checkmomentaliasing(
        testm, trialm, buffers, plans; trees=(test=ttree, trial=rtree)
    )
    @test !haserror(report.findings, :out_of_pool)
    @test report.ok
end

@testset "per-side moment sizing survives the call (rank 5 test / rank 3 trial)" begin
    # Part E Phase 5b step 0 regression.
    #
    # `storagemomentsplan` takes the test/trial moment types and sizes as four adjacent, UNTYPED
    # positional arguments in two pairs. Nothing in the type system pairs them, and
    # `_storagemomentsplan` dispatches on PLAN types rather than position, so it normalizes argument
    # order and would silently absorb a transposed pair introduced upstream.
    #
    # That mistake is invisible almost everywhere: Galerkin passes the same values for both sides, and
    # so did every Petrov caller until now, so a swap changes nothing. It becomes real exactly when the
    # two sides carry DIFFERENT moment sizes -- which is what Phase 5b introduces.
    #
    # So this test asks for genuinely different per-side ranks and checks each side got ITS OWN. It is
    # small, but it is currently the only thing that distinguishes correct per-side sizing from a swap.
    testpts = [SVector(0.05i, 0.0, 0.0) for i in 1:64]
    trialpts = [SVector(0.05i + 0.02, 0.0, 0.0) for i in 1:64]
    tree = H2Trees.buildtree(
        testpts,
        trialpts;
        builder=H2Trees.BlockTreeBuilder(;
            test=H2Trees.TwoNTreeBuilder(; minhalfsize=0.3, minvalues=0),
            trial=H2Trees.TwoNTreeBuilder(; minhalfsize=0.3, minvalues=0),
        ),
    )
    plans = H2Trees.buildplans(tree)

    # Asymmetric on purpose: if the pairs were transposed, test would come back (3,) and trial (5,).
    _, testmoments, trialmoments = H2Factory.storagemomentsplan(
        plans.testaggregationplan,
        plans.trialaggregationplan,
        tree;
        TestMomentType=(level) -> H2Factory.Moment{1,ComplexF64},
        TrialMomentType=(level) -> H2Factory.Moment{1,ComplexF64},
        testmomentsize=(tr, level) -> (5,),
        trialmomentsize=(tr, level) -> (3,),
    )

    @test all(size(m[1]) == (5,) for m in values(testmoments))
    @test all(size(m[1]) == (3,) for m in values(trialmoments))

    # The keyword form must agree with the positional form it delegates to -- otherwise the "safe"
    # entry point could itself be the thing that transposes the pair.
    _, testpos, trialpos = H2Factory.storagemomentsplan(
        plans.testaggregationplan,
        plans.trialaggregationplan,
        tree,
        (level) -> H2Factory.Moment{1,ComplexF64},
        (level) -> H2Factory.Moment{1,ComplexF64},
        (tr, level) -> (5,),
        (tr, level) -> (3,),
    )
    @test all(size(m[1]) == (5,) for m in values(testpos))
    @test all(size(m[1]) == (3,) for m in values(trialpos))
end

@testset "rectangular Toeplitz layout aliases soundly through the pooled allocator" begin
    # Part E Phase 5b (rectangular Toeplitz). The design fear was that the pooled allocator sizes each
    # side\'s toeplitz buffer LOCALLY and so could not handle a rectangular pair. This test refutes
    # that empirically: the allocator takes the toeplitz size `tsize` as an INDEPENDENT given (the
    # `tsize = standardtoeplitzsize(momentsize)` recompute in `unsafe_wraps.jl` is commented out) and
    # the aliasing checker reads the ACTUAL wrapped array sizes -- so a pair whose two sides have
    # DIFFERENT unpadded moment sizes but the SAME common `m+n-1` toeplitz buffer aliases cleanly with
    # no allocator change. The pair-dependency lives entirely in the caller\'s sizing functor.
    #
    # BlockTree so the two sides are genuinely separate pools (Petrov). Unpadded (4,) vs (6,); common
    # toeplitz (m+n-1) = (9,).
    testpts = [SVector(0.05i, 0.0, 0.0) for i in 1:64]
    trialpts = [SVector(0.05i + 2.0, 0.0, 0.0) for i in 1:64]
    bt = H2Trees.buildtree(
        testpts,
        trialpts;
        builder=H2Trees.BlockTreeBuilder(;
            test=H2Trees.TwoNTreeBuilder(; minhalfsize=0.3, minvalues=0),
            trial=H2Trees.TwoNTreeBuilder(; minhalfsize=0.3, minvalues=0),
        ),
    )
    plans = H2Trees.buildplans(bt)

    m, n = 4, 6
    P = m + n - 1
    buffers, testm, trialm = H2Factory.storagemomentsplan(
        plans.testaggregationplan,
        plans.trialaggregationplan,
        bt;
        TestMomentType=level -> H2Factory.ToeplitzMoment1D{ComplexF64},
        TrialMomentType=level -> H2Factory.ToeplitzMoment1D{ComplexF64},
        testmomentsize=(tr, level) -> ((m,), (P,)),
        trialmomentsize=(tr, level) -> ((n,), (P,)),
    )
    report = H2Factory.checkmomentaliasing(
        testm,
        trialm,
        buffers,
        plans;
        trees=(test=H2Trees.testtree(bt), trial=H2Trees.trialtree(bt)),
    )
    @test report.ok
    @test !haserror(report.findings, :out_of_pool)
    @test !haserror(report.findings, :stored_overlap)
    @test !haserror(report.findings, :same_level_overlap)
    @test !haserror(report.findings, :misaligned_extent)

    # NON-VACUITY: the two sides really do differ in unpadded size while sharing the toeplitz size --
    # otherwise this is just the square case with a mislabel.
    @test unique(size(mm.moments[1]) for mm in values(testm)) == [(m,)]
    @test unique(size(mm.moments[1]) for mm in values(trialm)) == [(n,)]
    @test unique(size(mm.toeplitzmoments[1]) for mm in values(testm)) == [(P,)]
    @test unique(size(mm.toeplitzmoments[1]) for mm in values(trialm)) == [(P,)]
end

end # module TestMomentAliasing
