module TestH2ToeplitzSmoke

using Test
using Random
using LinearAlgebra
using FFTW # concrete FFT backend for AbstractFFTs -- without one, `fft`/`plan_fft!` have no
# implementation to dispatch to and construction below throws a `MethodError`.
using AbstractFFTs
using H2Factory

# `ToeplitzMatrixRepresentation`/`AdjointToeplitzMatrixRepresentation` are core H2Factory types, but
# their `LinearMaps._unsafe_mul!` only exists once `ext/H2Toeplitz` loads (gated on `AbstractFFTs`).
# This file pins that matvec against an independent oracle.
#
# What the operator IS: the representation stores an ALREADY-FFT'd kernel of size `2s-1` per
# dimension, laid out in WRAP-AROUND (circulant-embedding) order. `_unsafe_mul!` zero-pads `x` to
# that size, multiplies in the FFT domain and truncates back to `s` -- which makes it exactly the
# multilevel Toeplitz operator
#
# Building that matrix densely is an oracle INDEPENDENT of the FFT path (no FFT, no shared code), so
# agreement pins the kernel convention, the padding, the truncation and the adjoint's conjugation.
# Note the wrap-around: with `ts = 2s-1` the transform is one sample too short for a linear
# convolution, so the wraparound is load-bearing, not incidental -- a naive linear-convolution
# reference does NOT match.
#
# Dense multilevel Toeplitz matrix implied by a wrap-around kernel.
function densemultileveltoeplitz(kernel, s)
    ts = size(kernel)
    lin = LinearIndices(s)
    T = zeros(eltype(kernel), prod(s), prod(s))
    for i in CartesianIndices(s), j in CartesianIndices(s)
        key = ntuple(d -> mod1(i[d] - j[d] + 1, ts[d]), length(s))
        T[lin[i], lin[j]] = kernel[key...]
    end
    return T
end

@testset "H2Toeplitz: FFT matvec == dense multilevel-Toeplitz oracle" begin
    rng = MersenneTwister(7)
    # Non-cubic shapes too: a bug that transposes/collapses the per-dimension sizes survives (2,2,2)
    # but not (3,2,4).
    for s in ((3,), (3, 2), (2, 2, 2), (3, 2, 2), (3, 3, 3), (3, 2, 4))
        ts = H2Factory.standardtoeplitzsize(s)
        kernel = randn(rng, ComplexF64, ts)
        A = H2Factory.ToeplitzMatrixRepresentation(fft(kernel))
        T = densemultileveltoeplitz(kernel, s)

        @test size(A) == (prod(s), prod(s))
        @test eltype(A) == ComplexF64
        @test H2Factory.toeplitzsize(A) == ts

        x = randn(rng, ComplexF64, prod(s))
        @test A * x ≈ T * x rtol = 1e-9

        # Adjoint: a separate code path (conjugate, swapped fft/ifft), so it needs its own check --
        # and against `T'`, not a re-derivation of it.
        Aadj = A'
        @test Aadj isa H2Factory.AdjointToeplitzMatrixRepresentation
        @test size(Aadj) == (prod(s), prod(s))
        @test Aadj * x ≈ T' * x rtol = 1e-9

        # Guard against a vacuous pass: the operator must not be ~zero.
        @test norm(T * x) > 1e-8
    end
end

@testset "H2Toeplitz: adjoint is the true adjoint (inner-product duality)" begin
    # Reference-free: <y, A*x> == <A'*y, x> must hold whatever the kernel convention is. Catches a
    # missing conjugation that a real-valued kernel would hide.
    rng = MersenneTwister(8)
    s = (3, 2, 2)
    kernel = randn(rng, ComplexF64, H2Factory.standardtoeplitzsize(s))
    A = H2Factory.ToeplitzMatrixRepresentation(fft(kernel))

    x = randn(rng, ComplexF64, prod(s))
    y = randn(rng, ComplexF64, prod(s))
    @test dot(y, A * x) ≈ dot(A' * y, x) rtol = 1e-9
end

end # module TestH2ToeplitzSmoke
