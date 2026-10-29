module TestRectangular

using Test
using Random
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils

include("component_harness_fixtures.jl")

# Part E PHASE 5 -- rectangular DOF coverage, driven with SIDE-OWNED transfers.
#
# The harness already had a rectangular Petrov config (256 test / 16 trial). What it did NOT have is
# rectangularity and side ownership in the SAME test: that config passes one shared transfer to both
# sides, so `trial i2i === test i2i` and the orientation flip is unobservable. Measured: with a
# shared transfer, deliberately breaking the flip leaves forward/transpose/adjoint at 1.8e-16 /
# 3.1e-16 / 3.1e-16 -- indistinguishable from correct.
#
# So this file forces BOTH sources of non-vacuity at once:
#   * rectangular size (256 x 16), so transpose/adjoint cannot hide row/column confusion;
#   * numerically DISTINCT per-side transfer constants, so resolving the wrong side changes the
#     ANSWER rather than only a label.
#
# THE TWO-CONSTANT CONSTRAINT (this is why there are two values, not four).
# `O2OTranslator` is constructed FROM the I2I translator, so `test O2O == transpose(test I2I)`
# structurally -- the two are not independent. Choosing four independent constants would break the
# transpose identity `transpose(A) == A^T` for reasons having nothing to do with side ownership, and
# the resulting failure would be misleading. The honest assignment is one constant per SIDE:
#
#     triali2i = trialo2o = C_TRIAL
#     testi2i  = testo2o  = C_TEST
#
# Then, per orientation:
#
#     forward    aggregation -> C_TRIAL   disaggregation -> C_TEST
#     transpose  aggregation -> C_TEST    disaggregation -> C_TRIAL     (mirrored)
#     adjoint    same side ownership as transpose, plus conjugation
#
# THE SIDE CONSTANTS ARE REAL. This is not incidental -- it is required, and finding out why was
# the substantive result of writing this file.
#
# The physics: I2I/O2O carry interpolation weights, which are REAL; the complex Green's-function kernel
# lives in the I2O coupling, which is where this fixture puts it (`C_I2O`). A complex LEVEL transfer is
# outside what this fixture is designed to exercise -- with real side constants every mode is exact.
#
# (Historical note: this was originally diagnosed under the old `conjmomentTransposeAdjoint!` moment-
# conjugation convention, since removed by the level-transfer refactor -- transpose/adjoint level
# transfers now apply `transpose(T)`/`adjoint(T)` explicitly at the transfer boundary via
# `leveltranslation!`. The real side constants used here are exact under either scheme; the removed
# convention's complex-transfer error figures are no longer meaningful.)
#
# Distinct REAL constants still make side loss numerically visible, which is all Phase 5 needs.
const C_TRIAL = 2.0 + 0.0im
const C_TEST = 3.0 + 0.0im
# Complex COUPLING, so `adjoint != transpose` is meaningful and the conjugation path is exercised.
const C_I2O = 0.75 - 1.25im

# Rectangular by construction: many more test DOFs than trial, matched `minvalues` so the asymmetry
# comes from point COUNT (robust) rather than mismatched `minvalues` (which can make `petrovplans`
# throw on non-contiguous disaggregation levels -- a known H2Trees constraint).
function rectangularmap(; ctrial=C_TRIAL, ctest=C_TEST, ntest::Int=256, ntrial::Int=16)
    return fakepetrovmap(
        mkpts(uniformpts(ntest)),
        mkpts(uniformpts(ntrial));
        T=ComplexF64,
        triali2i=scalartranslator(ctrial),
        trialo2o=scalartranslator(ctrial),
        testi2i=scalartranslator(ctest),
        testo2o=scalartranslator(ctest),
        i2o=scalartranslator(C_I2O),
        minhalfsize=0.004,
        testminvalues=1,
        trialminvalues=1,
    )
end

relerr(a, b) = norm(a - b) / max(norm(b), eps())

# Drive all three modes against the independently assembled complete matrix.
function checkallmodes(A, F; atol=1e-12)
    rng = MersenneTwister(20260720)
    x = randn(rng, ComplexF64, size(A, 2))
    y = randn(rng, ComplexF64, size(A, 1))

    yf = zeros(ComplexF64, size(A, 1))
    H2Factory.farmul!(yf, A, x)
    @test relerr(yf, F * x) < atol

    yt = zeros(ComplexF64, size(A, 2))
    H2Factory.farmul!(yt, transpose(A), y)
    @test relerr(yt, transpose(F) * y) < atol

    ya = zeros(ComplexF64, size(A, 2))
    H2Factory.farmul!(ya, adjoint(A), y)
    @test relerr(ya, adjoint(F) * y) < atol

    return nothing
end

@testset "Rectangular Petrov with side-owned transfers: all modes vs complete matrix" begin
    A = rectangularmap()

    # Rectangular, and genuinely so -- different DOF counts AND different tree depths.
    @test size(A) == (256, 16)
    @test leaflevels(H2Trees.testtree(A)) != leaflevels(H2Trees.trialtree(A))

    # The sides are numerically distinct, not merely differently named. Without this the numerical
    # assertions below would be satisfied by a map that ignores side ownership entirely.
    @test H2Factory.triali2itranslator(A) !== H2Factory.testi2itranslator(A)
    @test H2Factory.trialo2otranslator(A) !== H2Factory.testo2otranslator(A)

    # Structural side resolution, per orientation (the contract in `maps/accessors.jl`).
    @test H2Factory.i2itranslator(A, H2Factory.isForward()) ===
        H2Factory.triali2itranslator(A)
    @test H2Factory.i2itranslator(A, H2Factory.isTranspose()) ===
        H2Factory.testi2itranslator(A)
    @test H2Factory.o2otranslator(A, H2Factory.isForward()) ===
        H2Factory.testo2otranslator(A)
    @test H2Factory.o2otranslator(A, H2Factory.isTranspose()) ===
        H2Factory.trialo2otranslator(A)

    # Numerical: forward, transpose and adjoint against the complete matrix. `denseh2matrix`
    # assembles from the FORWARD factors via the per-side accessors (trial i2i for aggregation, test
    # o2o for disaggregation), so agreement here says the traversal applies the TRANSPOSE OF THE
    # FORWARD FACTORS in the reversed modes -- not forward kernels under swapped names.
    F = denseh2matrix(A)
    @test size(F) == (256, 16)
    checkallmodes(A, F)

    # Adjoint must genuinely differ from transpose, or the conjugation path is untested and the
    # adjoint assertion above would be satisfied by the transpose result.
    rng = MersenneTwister(1234)
    y = randn(rng, ComplexF64, size(A, 1))
    yt = zeros(ComplexF64, size(A, 2))
    ya = zeros(ComplexF64, size(A, 2))
    H2Factory.farmul!(yt, transpose(A), y)
    H2Factory.farmul!(ya, adjoint(A), y)
    @test norm(ya - yt) / norm(yt) > 0.1
end

@testset "Rectangular Petrov, reversed aspect ratio (16 x 256)" begin
    # The mirrored shape, so neither orientation is privileged by the fixture. A row/column confusion
    # that happens to be benign at 256x16 need not be at 16x256.
    A = rectangularmap(; ntest=16, ntrial=256)
    @test size(A) == (16, 256)
    F = denseh2matrix(A)
    @test size(F) == (16, 256)
    checkallmodes(A, F)
end

@testset "CONTROL: equal side constants make side loss numerically invisible" begin
    # This control exists to prove the tests above are sensitive FOR THE INTENDED REASON -- side
    # resolution -- rather than being generally brittle.
    #
    # Here both sides carry the SAME constant, which is the shape the pre-existing harness used. The
    # map is still rectangular, and all three modes still agree with the complete matrix, so this
    # config is a perfectly good NUMERICAL test. But it is blind to side ownership: with
    # `C_TRIAL == C_TEST` the two sides are numerically interchangeable, so a broken orientation flip
    # cannot change any number.
    #
    # Verified by mutation (2026-07-20), deliberately breaking the flip in `maps/accessors.jl`:
    #
    #   under the i2i-flip mutation     transpose err    adjoint err
    #   distinct constants              0.337            0.337      <- caught
    #   equal constants (this control)  8.3e-17          1.1e-16    <- invisible
    #
    # The o2o-flip mutation behaves the same way. In the distinct-constant testsets above, BOTH the
    # numerical assertions (transpose and adjoint) and the structural one fail under either mutation,
    # while the forward assertion correctly still passes -- forward is unaffected by a transposed-mode
    # side flip.
    #
    # That contrast is the evidence that the distinct-constant configuration adds real coverage.
    # `test_sideownership.jl` catches the same mutations structurally regardless of constants.
    c = 2.0 + 0.0im
    A = rectangularmap(; ctrial=c, ctest=c)

    @test size(A) == (256, 16)
    F = denseh2matrix(A)
    checkallmodes(A, F)

    # Numerically interchangeable, which is exactly what makes this configuration side-blind.
    @test H2Factory.triali2itranslator(A) == H2Factory.testi2itranslator(A)
end

end # module TestRectangular
