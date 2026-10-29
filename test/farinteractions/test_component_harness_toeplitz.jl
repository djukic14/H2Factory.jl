module TestComponentHarnessToeplitz

using Test
using Random
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils

include("component_harness_fixtures.jl")

@createtoeplitzmoment TestHarnessToeplitzMoment
@createmomentcollection TestHarnessToeplitzMomentCollection
@linkmomenttocollection TestHarnessToeplitzMoment TestHarnessToeplitzMomentCollection

const TEST_TOEPLITZ_TYPES = (;
    MomentType=TestHarnessToeplitzMoment,
    MomentCollectionType=TestHarnessToeplitzMomentCollection,
)

function checktesttoeplitzworkflow(A)
    @test all(m -> m isa TestHarnessToeplitzMoment, values(H2Factory.trialmoments(A)))
    @test all(m -> m isa TestHarnessToeplitzMoment, values(H2Factory.testmoments(A)))
    # Both sides, always: for a Galerkin map the two accessors resolve to the same shared
    # dictionary, so this is the Petrov check and the Galerkin one at once -- and it no longer
    # depends on which FIELDS the map happens to have, which a workspace does not expose.
    @test all(
        m -> m isa TestHarnessToeplitzMomentCollection,
        values(H2Factory.trialmomentcollections(A)),
    )
    @test all(
        m -> m isa TestHarnessToeplitzMomentCollection,
        values(H2Factory.testmomentcollections(A)),
    )
    return A
end

testfaketoeplitzmap(args...; kwargs...) = checktesttoeplitzworkflow(
    faketoeplitzmap(args...; merge((; kwargs...), TEST_TOEPLITZ_TYPES)...)
)

testfakepetrovtoeplitzmap(args...; kwargs...) = checktesttoeplitzworkflow(
    fakepetrovtoeplitzmap(args...; merge((; kwargs...), TEST_TOEPLITZ_TYPES)...)
)

function testminitoeplitzh2matrix(args...; kwargs...)
    m = minitoeplitzh2matrix(args...; merge((; kwargs...), TEST_TOEPLITZ_TYPES)...)
    checktesttoeplitzworkflow(m.A)
    return m
end

function testminipetrovtoeplitzh2matrix(args...; kwargs...)
    m = minipetrovtoeplitzh2matrix(args...; merge((; kwargs...), TEST_TOEPLITZ_TYPES)...)
    checktesttoeplitzworkflow(m.A)
    return m
end

# Toeplitz analogue of `test_component_harness.jl`. Where that harness plugs dense/scalar i2o
# translators into the fake Galerkin map, this drives H2Factory's REAL FFT Toeplitz far-field path
# (modulate_in! -> toeplitzmul! -> modulate_out!) through the FULL multi-level
# aggregate/translate/disaggregate traversal -- not just the single-pair i2o step `checktoeplitzi2o`
# covers. The map is a `faketoeplitzmap` (ToeplitzMoments + a real `ToeplitzI2OTranslationCollection`
# + `ToeplitzI2OChannel`), so every dense-fake oracle (`ref_farmul`, `denseh2matrix`, spy records)
# applies unchanged. Storage is always ComplexF64 (the FFT path is complex-storage); "real vs complex"
# is a property of the KERNEL's values (real kernel: adjoint == transpose; complex: adjoint != transpose).
Random.seed!(0x2c0d_ab12)

@testset "Reference oracle agreement (ref_farmul == farmul!, real + complex kernel)" begin
    for (kname, kernel) in toeplitz_kernel_cases()
        tree, N = b4tree()
        A = testfaketoeplitzmap(tree, kernel; rank=2)
        x = randn(ComplexF64, N)
        @test ref_farmul(A, x) ≈ H2Factory.farmul!(similar(x), A, x)

        A2 = testfaketoeplitzmap(
            H2Trees.buildtree(
                [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
                builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
            ),
            kernel;
            rank=3,
        )
        x2 = randn(ComplexF64, 256)
        @test ref_farmul(A2, x2) ≈ H2Factory.farmul!(similar(x2), A2, x2)
    end
end

@testset "Complete dense matrix (denseh2matrix): mul! all modes, real + complex, uniform + non-uniform" begin
    # The Toeplitz mul reproduces the dense coupling S_ts, so `denseh2matrix(A)` (assembled from the
    # exact couplings, independent of mul!) equals the map. `mul!` in ALL three modes must match F /
    # transpose(F) / adjoint(F), for both a real kernel (adjoint == transpose) and a complex one
    # (adjoint != transpose), across both aggregation modes.
    for (kname, kernel) in toeplitz_kernel_cases()
        trees = Dict(
            "uniform" => (
                H2Trees.buildtree(
                    [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
                    builder=H2Trees.TwoNTreeBuilder(;
                        minhalfsize=0.008, minvalues=0, root=1
                    ),
                ),
                256,
            ),
            "non-uniform" => nonuniformtree(),
        )
        for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
            for (name, (tree, N)) in trees
                A = testfaketoeplitzmap(tree, kernel; rank=2, aggregationmode=mode)
                F = denseh2matrix(A)
                x = randn(ComplexF64, N)
                @testset "$kname / $name / $(typeof(mode))" begin
                    @test H2Factory.farmul!(similar(x), A, x) ≈ F * x
                    @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x
                    @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ F' * x
                    @test ref_farmul(A, x) ≈ F * x
                    # Per-pair I2O translation (real FFT single-pair pipeline, all modes) vs the
                    # dense Toeplitz coupling.
                    @test checktoeplitzi2o(tree, kernel, 2; aggregationmode=mode, n=3) > 0
                end
            end
        end
    end
    @test length(leaflevels(nonuniformtree()[1])) > 1
end

@testset "Aggregation makes sense (per-node moments == reference)" begin
    tree, N = b4tree()
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=3)
    x = randn(ComplexF64, N)
    H2Factory.aggregate!(A, x)
    refagg = ref_aggregate(A, x)
    for (node, refm) in refagg
        # Moment content is a (rank,1,1) tensor; the reference works in length-rank vectors.
        @test vec(H2Factory.trialmoments(A)[node][1]) ≈ refm
    end
end

@testset "Aggregation closed form (identity-sum -> subtree sums, incl. top)" begin
    # Identity I2I + all-ones rank leaf basis -> every aggregated node's moment is the sum of x over
    # its subtree DOFs (each of the `rank` components equals that sum). AggregateMode leaves the
    # spatial moment un-modulated after `aggregate!`, so this reads the plain aggregated content.
    tree, N = b4tree()
    A = testfaketoeplitzmap(tree, z -> 1.0 + 0.0im; rank=2)
    x = randn(ComplexF64, N)
    H2Factory.aggregate!(A, x)
    plan = H2Factory.trialaggregationplan(A)
    for level in H2Trees.aggregationlevels(plan)
        for node in H2Trees.aggregationnodes(plan, level)
            @test H2Factory.trialmoments(A)[node][1][1] ≈ sum(x[H2Trees.values(tree, node)])
        end
    end
    toplevel = last(collect(H2Trees.aggregationlevels(plan)))
    topnodes = collect(H2Trees.aggregationnodes(plan, toplevel))
    @test sum(H2Factory.trialmoments(A)[n][1][1] for n in topnodes) ≈ sum(x)
end

@testset "Disaggregation makes sense (aggregate! -> disaggregate! == reference)" begin
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(128)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=2)
    x = randn(ComplexF64, 128)
    H2Factory.aggregate!(A, x)
    y = zeros(ComplexF64, 128)
    H2Factory.disaggregate!(y, A)
    agg = ref_aggregate(A, x)
    yref = ref_disaggregate(A, ref_translate(A, agg))
    @test y ≈ yref
end

@testset "Transpose/adjoint duality (reference-free)" begin
    # <u, A x> == <A' u, x> over ℂ (adjoint, not transpose), on the deep tree with a complex kernel.
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=2)
    x = randn(ComplexF64, 256)
    u = randn(ComplexF64, 256)
    lhs = dot(u, H2Factory.farmul!(similar(x), A, x))
    rhs = dot(H2Factory.farmul!(similar(u), adjoint(A), u), x)
    @test lhs ≈ rhs
end

@testset "Structural correctness (spy translator + i2o records)" begin
    tree, N = b4tree()
    i2ispy = spytranslator(identitytranslator())
    A = testfaketoeplitzmap(
        tree, toeplitz_complexkernel; rank=2, i2i=i2ispy, o2o=identitytranslator()
    )
    x = randn(ComplexF64, N)
    H2Factory.farmul!(similar(x), A, x)

    # The FakeToeplitzI2O wrapper records every i2o/o2i call. Every far pair translated exactly once,
    # no near pair ever translated.
    farset = Set(ref_farpairs(A))
    i2orecs = filter(r -> r.role == :i2o, records(H2Factory.i2otranslator(A)))
    @test length(i2orecs) == length(farset)
    @test Set((r.a, r.b) for r in i2orecs) == farset

    # Every non-leaf aggregated strictly after all its children (i2i transfer ordering).
    transfers = filter(r -> r.role == :transfer, records(i2ispy))
    for (idx, r) in enumerate(transfers)
        childasparent = findall(s -> s.a == r.b, transfers)
        @test all(<(idx), childasparent)
    end
end

@testset "Petrov (BlockTree) Toeplitz: all modes vs complete matrix; balanced + unbalanced + asymmetric" begin
    # SEPARATE test/trial trees; aggregation on the trial side, disaggregation on the test side, so
    # F is Ntest × Ntrial. The real FFT Toeplitz traversal must match the independently-assembled
    # complete matrix in ALL modes -- across coincident / disjoint-far / overlapping geometries, both
    # balanced (uniform) and unbalanced (dense cluster + isolated points) trees, and asymmetric DOF
    # counts (different test/trial tree depths). Same geometries as the dense Petrov harness.
    u = uniformpts(64)
    # BOTH aggregation modes on the Petrov side (AggregateMode: translation in disaggregation;
    # AggregateTranslateMode: translation in aggregation) -- exercising the asymmetric test/trial
    # BlockTree traversal under each, which neither this nor the dense Petrov harness did before.
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        for (kname, kernel) in toeplitz_kernel_cases()
            for (name, tp, rp, tv, rv, mhs) in toeplitz_petrov_point_configs()
                A = testfakepetrovtoeplitzmap(
                    tp,
                    rp,
                    kernel;
                    rank=2,
                    minhalfsize=mhs,
                    testminvalues=tv,
                    trialminvalues=rv,
                    aggregationmode=mode,
                )
                F = denseh2matrix(A)
                x = randn(ComplexF64, size(A, 2))
                w = randn(ComplexF64, size(A, 1))
                @testset "$(typeof(mode)) / $kname / $name" begin
                    @test H2Factory.farmul!(similar(x, size(A, 1)), A, x) ≈ F * x
                    @test H2Factory.farmul!(similar(w, size(A, 2)), transpose(A), w) ≈
                        transpose(F) * w
                    @test H2Factory.farmul!(similar(w, size(A, 2)), adjoint(A), w) ≈ F' * w
                    @test ref_farmul(A, x) ≈ F * x
                    @test checktoeplitzi2o(
                        H2Trees.tree(A), kernel, 2; aggregationmode=mode, n=3
                    ) > 0
                end
            end
        end
    end
    # disjoint-far closed form: entirely-far regions -> a single root<->root far pair; the far matrix
    # is rank-1 in the coupling, F = S-block broadcast (not necessarily ones -- the Toeplitz coupling
    # is the kernel matrix), so just confirm non-emptiness + symmetry structure via the matrix checks
    # above. overlapping really is non-symmetric:
    Aov = testfakepetrovtoeplitzmap(
        mkpts(u), mkpts(0.5 .+ u), toeplitz_complexkernel; rank=2, minhalfsize=0.02
    )
    @test !issymmetric(denseh2matrix(Aov))
    # many-test genuinely rectangular + different tree depths.
    Amany = testfakepetrovtoeplitzmap(
        mkpts(uniformpts(128)),
        mkpts(uniformpts(16)),
        toeplitz_complexkernel;
        rank=2,
        minhalfsize=0.008,
        testminvalues=1,
        trialminvalues=1,
    )
    @test size(Amany) == (128, 16)
end

@testset "Sampled kernel exact (interpolation mini vs K[i,j]=kernel(x_i-x_j))" begin
    # THE accuracy oracle. `faketoeplitzmap` (identity transfers) only proves the machinery is
    # self-consistent (farmul! == denseh2matrix for ANY kernel). To prove the Toeplitz path computes
    # the right PHYSICS, `minitoeplitzh2matrix` is a genuine interpolation FMM (real M2M/L2L transfers)
    # whose far field reproduces the ACTUALLY-SAMPLED kernel K[i,j]=kernel(x_i-x_j). For a kernel that
    # degree-(rank-1) interpolation is exact for -- an AFFINE k(z)=a+b*z at rank 2 -- the far field
    # equals K exactly, in ALL modes (the Toeplitz analogue of the dense mini's bilinear-exact test).
    pts = uniformpts(128)
    for (kname, a, b) in (
        ("real affine", 1.0 + 0.0im, 0.5 + 0.0im),
        ("complex affine", 1.0 + 0.3im, 0.4 - 0.2im),
    )
        kern(z) = a + b * z
        m = testminitoeplitzh2matrix(pts, kern; rank=2, minhalfsize=0.02)
        A = m.A
        K = m.dense_true
        x = randn(ComplexF64, length(pts))
        @testset "$kname" begin
            @test m.dense_approx ≈ K                                  # interpolation exact -> Ã == sampled K
            @test H2Factory.farmul!(similar(x), A, x) ≈ K * x         # forward matches the sampled kernel
            @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(K) * x
            @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ K' * x
            @test H2Factory.farmul!(similar(x), A, x) ≈ m.dense_approx * x   # machinery (Ã) too

            for α in [false, true]
                for β in [false, true]
                    y = randn(eltype(A), size(A, 1))
                    y2 = deepcopy(y)

                    @test LinearAlgebra.mul!(y, A, x, α, β) ≈
                        LinearAlgebra.mul!(y2, K, x, α, β)

                    w = randn(eltype(A), size(A, 2))
                    w2 = deepcopy(w)

                    @test LinearAlgebra.mul!(w, transpose(A), y, α, β) ≈
                        LinearAlgebra.mul!(w2, transpose(K), y, α, β)

                    w2 = deepcopy(w)

                    @test LinearAlgebra.mul!(w, adjoint(A), y, α, β) ≈
                        LinearAlgebra.mul!(w2, K', y, α, β)
                end
            end
        end
    end

    # Approximation quality vs the TRUE kernel: for a general smooth (non-polynomial) kernel, higher
    # interpolation order must reduce the error -- the classic order-of-accuracy improvement, now
    # through the FFT Toeplitz path.
    smooth(z) = exp(im * z) / (1 + z^2)
    x = randn(ComplexF64, length(pts))
    e2 = let m = testminitoeplitzh2matrix(pts, smooth; rank=2, minhalfsize=0.02)
        norm((m.dense_true - m.dense_approx) * x) / norm(m.dense_true * x)
    end
    e3 = let m = testminitoeplitzh2matrix(pts, smooth; rank=3, minhalfsize=0.02)
        norm((m.dense_true - m.dense_approx) * x) / norm(m.dense_true * x)
    end
    @test e3 < e2

    # Both aggregation modes reproduce the sampled affine kernel (machinery is mode-agnostic).
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        m = testminitoeplitzh2matrix(
            pts,
            z -> 2.0 - 1.0im + (0.5 + 0.25im) * z;
            rank=2,
            minhalfsize=0.02,
            aggregationmode=mode,
        )
        xx = randn(ComplexF64, length(pts))
        @test H2Factory.farmul!(similar(xx), m.A, xx) ≈ m.dense_true * xx
    end
end

@testset "Sampled kernel exact -- Petrov (interpolation mini vs K[i,j]=kernel(x_test_i - x_trial_j))" begin
    # The Petrov analogue of the sampled-kernel oracle: a genuine interpolation FMM with SEPARATE
    # test/trial trees and REAL transfers on each side, whose far field reproduces the ACTUALLY-
    # SAMPLED kernel K[i,j]=kernel(x_test_i - x_trial_j). Neither this harness nor the dense one
    # checked the asymmetric Petrov path against the true kernel before (only against denseh2matrix) --
    # and that path is exactly where the known iFMM Petrov bug lived, so it is the highest-value check
    # here. Covers same points (coincident), overlapping, disjoint-far, and asymmetric test/trial
    # depths, under BOTH aggregation modes, for real and complex affine kernels (rank-2 exact).
    u = uniformpts(64)
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        for (kname, kern) in toeplitz_affine_kernel_cases()
            for (name, tp, rp, tv, rv, mhs) in toeplitz_petrov_coordinate_configs()
                m = testminipetrovtoeplitzh2matrix(
                    tp,
                    rp,
                    kern;
                    rank=2,
                    minhalfsize=mhs,
                    testminvalues=tv,
                    trialminvalues=rv,
                    aggregationmode=mode,
                )
                A = m.A
                K = m.dense_true
                x = randn(ComplexF64, size(A, 2))
                w = randn(ComplexF64, size(A, 1))
                @testset "$(typeof(mode)) / $kname / $name" begin
                    @test m.dense_approx ≈ K                                 # Ã == sampled K (exact)
                    @test H2Factory.farmul!(similar(x, size(A, 1)), A, x) ≈ K * x
                    @test H2Factory.farmul!(similar(w, size(A, 2)), transpose(A), w) ≈
                        transpose(K) * w
                    @test H2Factory.farmul!(similar(w, size(A, 2)), adjoint(A), w) ≈ K' * w

                    for α in [false, true]
                        for β in [false, true]
                            y = randn(eltype(A), size(A, 1))
                            y2 = deepcopy(y)

                            @test LinearAlgebra.mul!(y, A, x, α, β) ≈
                                LinearAlgebra.mul!(y2, K, x, α, β)

                            w = randn(eltype(A), size(A, 2))
                            w2 = deepcopy(w)

                            @test LinearAlgebra.mul!(w, transpose(A), y, α, β) ≈
                                LinearAlgebra.mul!(w2, transpose(K), y, α, β)

                            w2 = deepcopy(w)

                            @test LinearAlgebra.mul!(w, adjoint(A), y, α, β) ≈
                                LinearAlgebra.mul!(w2, K', y, α, β)
                        end
                    end
                end
            end
        end
    end
    # Approximation quality vs the TRUE kernel on a genuinely non-symmetric (overlapping) Petrov
    # geometry: higher interpolation order reduces the error.
    smooth(z) = exp(im * z) / (1 + z^2)
    e2 =
        let m = testminipetrovtoeplitzh2matrix(
                u, 0.5 .+ u, smooth; rank=2, minhalfsize=0.02
            )
            xx = randn(ComplexF64, size(m.A, 2))
            norm((m.dense_true - m.dense_approx) * xx) / norm(m.dense_true * xx)
        end
    e3 =
        let m = testminipetrovtoeplitzh2matrix(
                u, 0.5 .+ u, smooth; rank=3, minhalfsize=0.02
            )
            xx = randn(ComplexF64, size(m.A, 2))
            norm((m.dense_true - m.dense_approx) * xx) / norm(m.dense_true * xx)
        end
    @test e3 < e2
end

@testset "Determinism (real traversal == single-threaded reference; bit-stable repeats)" begin
    tree = H2Trees.buildtree(
        [SVector(c, 0.0, 0.0) for c in uniformpts(512)];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
    )
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=2)
    x = randn(ComplexF64, 512)
    y1 = similar(x)
    y2 = similar(x)
    H2Factory.farmul!(y1, A, x)
    H2Factory.farmul!(y2, A, x)
    @test y1 == y2
    @test y1 ≈ ref_farmul(A, x)
end

@testset "FakeToeplitzI2O forwards its modulation policy (regression)" begin
    # `FakeToeplitzI2O` forwards `assertnosymmetry`/`i2otrait` to its wrapped collection already;
    # `modulation` must be forwarded the same way, or `I2OTranslation(wrapper)`'s auto-selection
    # (`modulation(coupling)`) silently resolves to the generic `NoModulation()` meant for ordinary
    # (non-Toeplitz) collections instead of the wrapped collection's `ToeplitzModulation()` -- and
    # the mandatory per-node FFT into the frequency domain is skipped with no error, not even a
    # wrong number close enough to look plausible (independently reproduced as 0.77 relative error
    # vs the dense reference, against 9.75e-17 with the delegation in place).
    tree, N = b4tree()
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=2)
    coup = H2Factory.coupling(H2Factory.i2otranslator(A))
    @test coup isa H2FactoryTestUtils.FakeToeplitzI2O
    @test H2Factory.modulation(coup) isa H2Factory.ToeplitzModulation
    @test H2Factory.modulation(H2Factory.I2OTranslation(coup)) isa
        H2Factory.ToeplitzModulation

    x = randn(ComplexF64, N)
    @test H2Factory.farmul!(similar(x), A, x) ≈ ref_farmul(A, x)
end

@testset "a Toeplitz moment paired with NoModulation rejects through a wrapper, not just the bare collection (regression)" begin
    # `modulate_in!`/`modulate_out!` reject a Toeplitz moment paired with `NoModulation` by
    # dispatching on the MOMENT's type ADDITIONALLY to the coupling's -- not instead of it (see
    # the "bare collection" testsets below for the original, coupling-typed rejection, which still
    # holds). The moment-typed rejection exists specifically so this also holds through
    # `FakeToeplitzI2O` -- or any other wrapper -- which the coupling-typed rejection alone can
    # never see. This is the entry point with the actual numerical risk: silently skipping the
    # frequency-domain conversion, not merely constructing a translation object with the wrong
    # policy on record.
    #
    # `I2OTranslation(coupling, modulation)`'s two-argument constructor, by contrast, is left
    # nominal in THIS change -- intentionally scoped out, not asserted incompatible with further
    # hardening. iFMM's B-spline routes use that constructor to compose a coupling with a
    # modulation other than its own default (`I2OTranslation(collection, collection.modulation)`),
    # which only rules out a BLANKET "chosen modulation must equal `modulation(coupling)`" check
    # (B-spline routes never pass that default). A NARROWER check rejecting specifically
    # `NoModulation()` for a coupling that needs Toeplitz modulation would not conflict with
    # B-spline usage at all; it is simply not implemented here. This test documents the gap as it
    # stands, not a reason it cannot be closed.
    tree, _ = b4tree()
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=2)
    coup = H2Factory.coupling(H2Factory.i2otranslator(A))
    @test coup isa H2FactoryTestUtils.FakeToeplitzI2O

    moment = first(values(H2Factory.testmoments(A)))
    @test_throws ArgumentError H2Factory.modulate_in!(
        moment,
        1,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        coup,
        nothing,
        nothing,
        nothing,
    )
    @test_throws ArgumentError H2Factory.modulate_out!(
        moment,
        1,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        coup,
        nothing,
        nothing,
        nothing,
    )

    # Known, documented gap: the two-argument constructor does not reject the same combination
    # through this wrapper (see comment above). Pinned here so a future fix to this constructor
    # is a deliberate, visible change to this test, not a silent behavior change.
    translation = H2Factory.I2OTranslation(coup, H2Factory.NoModulation())
    @test H2Factory.modulation(translation) isa H2Factory.NoModulation
end

@testset "a Toeplitz moment paired with NoModulation rejects through the bare collection too (regression)" begin
    # The ORIGINAL rejection boundary: a Toeplitz moment reaching a BARE
    # `ToeplitzI2OTranslationCollection` (no wrapper at all) paired with `NoModulation` must still
    # reject. Both the coupling-typed rejection (`toeplitzi2ocollection.jl`) and the moment-typed
    # one (`i2oprotocol.jl`, added for the wrapper case above) apply here at once; an intersection
    # method resolves the resulting dispatch ambiguity. Exercising this directly guards against a
    # future change to either rejection silently dropping the other's coverage.
    tree, _ = b4tree()
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=2)
    coup = H2Factory.coupling(H2Factory.i2otranslator(A))
    @test coup isa H2FactoryTestUtils.FakeToeplitzI2O
    barecollection = coup.collection
    @test barecollection isa H2Factory.ToeplitzI2OTranslationCollection

    toeplitzmoment = first(values(H2Factory.testmoments(A)))
    @test_throws ArgumentError H2Factory.modulate_in!(
        toeplitzmoment,
        1,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        barecollection,
        nothing,
        nothing,
        nothing,
    )
    @test_throws ArgumentError H2Factory.modulate_out!(
        toeplitzmoment,
        1,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        barecollection,
        nothing,
        nothing,
        nothing,
    )
end

@testset "an ordinary moment paired with NoModulation rejects through a bare Toeplitz collection (regression)" begin
    # The case the earlier moment-typed-only rejection would have missed: an ORDINARY (non-Toeplitz)
    # moment reaching a bare `ToeplitzI2OTranslationCollection`. If the coupling itself is
    # Toeplitz-shaped, pairing it with `NoModulation` is wrong regardless of which moment type
    # happens to flow through it -- the coupling-typed rejection in `toeplitzi2ocollection.jl`
    # covers exactly this, independent of the moment-typed one added for the wrapper case.
    tree, _ = b4tree()
    A = testfaketoeplitzmap(tree, toeplitz_complexkernel; rank=2)
    coup = H2Factory.coupling(H2Factory.i2otranslator(A))
    barecollection = coup.collection
    @test barecollection isa H2Factory.ToeplitzI2OTranslationCollection

    ordinarymoment = H2Factory.Moment{2,Float64}((3, 3))
    @test_throws ArgumentError H2Factory.modulate_in!(
        ordinarymoment,
        1,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        barecollection,
        nothing,
        nothing,
        nothing,
    )
    @test_throws ArgumentError H2Factory.modulate_out!(
        ordinarymoment,
        1,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        barecollection,
        nothing,
        nothing,
        nothing,
    )
end

end # module TestComponentHarnessToeplitz
