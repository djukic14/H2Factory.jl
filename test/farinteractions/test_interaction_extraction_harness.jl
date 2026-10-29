module TestInteractionExtractionHarness

using Test
using LinearAlgebra
using SparseArrays
using StaticArrays
using H2Factory
using H2Trees
using H2FactoryTestUtils

include("component_harness_fixtures.jl")

function checkkernelpartition(A, K, farref)
    far = H2Factory.farinteractions(K, A)
    near = H2Factory.nearinteractions(K, A)
    nearfull = Matrix(near)

    @test maximum(abs, far - farref) < 1e-10
    @test maximum(abs, nearfull - (K - farref)) < 1e-10
    @test maximum(abs, nearfull + far - K) < 1e-10
    @test all(iszero, nearfull .* far)

    return true
end

@testset "Galerkin real/complex exact-kernel extraction: near + far partition the full matrix" begin
    configs = (
        ("b4", b4points(), 2.0, H2Trees.AggregateMode()),
        ("uniform / aggregate", uniformpts(256), 0.008, H2Trees.AggregateMode()),
        (
            "uniform / aggregate-translate",
            uniformpts(256),
            0.008,
            H2Trees.AggregateTranslateMode(),
        ),
        ("non-uniform / aggregate", nonuniformpoints(), 0.0005, H2Trees.AggregateMode()),
        (
            "non-uniform / aggregate-translate",
            nonuniformpoints(),
            0.0005,
            H2Trees.AggregateTranslateMode(),
        ),
        (
            "many separated clusters",
            [p[1] for p in first(manyclusters(6, 20))],
            1.0,
            H2Trees.AggregateMode(),
        ),
    )
    for (kname, kernel) in affine_kernel_cases()
        for (name, points, minhalfsize, mode) in configs
            mini = minih2matrix(
                points; rank=2, kernel=kernel, minhalfsize=minhalfsize, aggregationmode=mode
            )
            K = fullkernelmatrix(mini.points, mini.points, kernel, eltype(mini.A))
            @testset "$kname / $name / $(typeof(mode))" begin
                @test mini.dense_approx ≈ mini.dense_true
                @test checkkernelpartition(mini.A, K, mini.dense_approx)
            end
        end
    end
end

@testset "Petrov real/complex exact-kernel extraction: near + far partition the full matrix" begin
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        for (kname, kernel) in affine_kernel_cases()
            for (name, testpts, trialpts, testminvalues, trialminvalues, minhalfsize) in
                dense_petrov_coordinate_configs()
                mini = minipetrovh2matrix(
                    testpts,
                    trialpts;
                    kernel=kernel,
                    minhalfsize=minhalfsize,
                    testminvalues=testminvalues,
                    trialminvalues=trialminvalues,
                    aggregationmode=mode,
                )
                K = fullkernelmatrix(
                    mini.testpoints, mini.trialpoints, kernel, eltype(mini.A)
                )
                @testset "$(typeof(mode)) / $kname / $name" begin
                    @test mini.dense_approx ≈ mini.dense_true
                    @test checkkernelpartition(mini.A, K, mini.dense_approx)
                end
            end
        end
    end
end

# Extraction under transpose/adjoint, which the forward-only testsets above never reached.
#
# Both `farinteractions` and `nearinteractions` resolve a `BlockTree` map's row/column sides, and
# both originally did so from the RAW tree (always forward) rather than through the map's
# orientation-aware `testtree`/`trialtree` accessors. The failure mode is nasty in the balanced
# case: with equal test/trial DOF counts the swap produces a silently WRONG matrix (measured ~1.25
# absolute error here) and raises nothing at all. It only becomes loud -- a thrown
# `TaskFailedException` -- once the two sides have different DOF counts. So this testset
# deliberately sweeps both the balanced configs (to catch the silent case) and the rectangular
# ones (to catch the throwing case), in both transposed orientations.
@testset "Petrov extraction under transpose/adjoint" begin
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        for (kname, kernel) in affine_kernel_cases()
            for (name, testpts, trialpts, testminvalues, trialminvalues, minhalfsize) in
                dense_petrov_coordinate_configs()
                mini = minipetrovh2matrix(
                    testpts,
                    trialpts;
                    kernel=kernel,
                    minhalfsize=minhalfsize,
                    testminvalues=testminvalues,
                    trialminvalues=trialminvalues,
                    aggregationmode=mode,
                )
                K = fullkernelmatrix(
                    mini.testpoints, mini.trialpoints, kernel, eltype(mini.A)
                )
                @testset "$(typeof(mode)) / $kname / $name" begin
                    for (opname, op) in (("transpose", transpose), ("adjoint", adjoint))
                        @testset "$opname" begin
                            @test checkkernelpartition(
                                op(mini.A), op(K), op(mini.dense_approx)
                            )
                            # The shape must actually flip -- a rows/cols swap that happens to
                            # stay square would otherwise slip through the numeric checks.
                            @test size(H2Factory.farinteractions(op(K), op(mini.A))) ==
                                reverse(size(mini.A))
                        end
                    end
                end
            end
        end
    end
end

# The `checkadmissibility(A)` convenience wrapper (H2Factory.Diagnostics -> H2Trees). It reads the
# plans in FORWARD orientation deliberately: admissibility is a property of the assembled scheme,
# not of the orientation it is applied in, so a map and its transpose/adjoint describe the same
# geometry and MUST report identically. Asserting that here is what pins the `forwardmap` choice --
# without it, an orientation-sensitive rewrite would go unnoticed.
@testset "checkadmissibility wrapper is orientation-independent" begin
    for (name, testpts, trialpts, testminvalues, trialminvalues, minhalfsize) in
        dense_petrov_coordinate_configs()
        mini = minipetrovh2matrix(
            testpts,
            trialpts;
            kernel=first(affine_kernel_cases())[2],
            minhalfsize=minhalfsize,
            testminvalues=testminvalues,
            trialminvalues=trialminvalues,
            aggregationmode=H2Trees.AggregateMode(),
        )
        @testset "$name" begin
            reports = [
                H2Factory.Diagnostics.admissibilityreport(M; coverage=false, throw=false)
                for M in (mini.A, transpose(mini.A), adjoint(mini.A))
            ]
            # A synthetic mini map is built on one consistent predicate, so it must be admissible.
            @test all(r -> r.ok, reports)
            @test allequal(length(r.findings) for r in reports)
        end
    end
end

end # module TestInteractionExtractionHarness
