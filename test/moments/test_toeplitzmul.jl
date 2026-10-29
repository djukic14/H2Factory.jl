module TestToeplitzMul

using Test
using Random
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory

# `toeplitzmul!` applies an already-FFT'd translation operator to an already-FFT'd moment, so in this
# domain it is a plain ELEMENTWISE (Hadamard) product -- no FFT oracle needed. That makes the whole
# α/β contract testable directly against a two-line reference:
#
#   scalar tf : tout[i] = β*tout[i] + α*(tf .* tin[i])           for each i
#   matrix tf : tout[i] = β*tout[i] + α*Σ_j tf[i,j] .* tin[j]    (D×D block matrix)
#   vector tf : the D1==1 case contracts D2 inputs into ONE output:
#               tout[1] = β*tout[1] + α*Σ_i tf[i] .* tin[i]
#               the D2==1 case broadcasts ONE input to D1 outputs:
#               tout[i] = β*tout[i] + α*(tf[i] .* tin[1])
#
# Only the `toeplitzmoments` field participates -- `moments` is untouched scratch here.

@createtoeplitzmoment TestToeplitzMulMoment

const MSIZE = (3, 3)

# Build a moment whose toeplitzmoments are filled with reproducible random values.
function randmoment(::Type{T}, D, rng) where {T}
    m = TestToeplitzMulMoment{D,T}(MSIZE)
    for i in 1:D
        H2Factory.toeplitzmoment(m, i) .= randn(
            rng, T, H2Factory.standardtoeplitzsize(MSIZE)
        )
    end
    return m
end

tmoments(m) = [copy(H2Factory.toeplitzmoment(m, i)) for i in eachindex(m)]

randtf(::Type{T}, dims...; rng) where {T} = map(
    _ -> H2Factory.ToeplitzMatrixRepresentation(
        randn(rng, T, H2Factory.standardtoeplitzsize(MSIZE))
    ),
    zeros(Int, dims...),
)

@testset verbose = true "toeplitzmul!" begin
    @testset "toeplitzmul!: matrix tf obeys the α/β contract" begin
        # A D×D block matrix of translation operators contracting D inputs into D outputs. The
        # `α=true, β=false` case is the one that matters: it is the plain `out = A*in` overwrite, and it
        # is what the aggregation path uses on the FIRST translating partner of a receiving node (nothing
        # to accumulate onto yet). A bug here silently zeroes that first contribution.
        rng = MersenneTwister(20)
        T = ComplexF64
        D = 3
        tf = randtf(T, D, D; rng=rng)
        in = randmoment(T, D, rng)
        tin = tmoments(in)

        for α in (true, false), β in (true, false)
            out = randmoment(T, D, rng)
            tout0 = tmoments(out)

            H2Factory.toeplitzmul!(out, tf, in, α, β, SerialScheduler())

            for i in 1:D
                expected = (β ? tout0[i] : zero(tout0[i]))
                if α
                    for j in 1:D
                        expected = expected .+ tf[i, j].toeplitz .* tin[j]
                    end
                end
                @test H2Factory.toeplitzmoment(out, i) ≈ expected
            end
            # `in` must never be mutated -- it is a shared, still-needed moment in the real traversal.
            for i in 1:D
                @test H2Factory.toeplitzmoment(in, i) == tin[i]
            end
        end
    end

    @testset "toeplitzmul!: vector tf, D1==1 contracts all inputs into one output" begin
        # `out` has a single component; `in` has D2. All D2 translation operators write into the SAME
        # output slot, so the result must be their SUM. This is the shape the scalar-test/vector-trial
        # (D1==1) mixed-dimension case takes, e.g. a scalar test space against a vector trial space.
        rng = MersenneTwister(21)
        T = ComplexF64
        D2 = 3
        tf = randtf(T, D2; rng=rng)
        in = randmoment(T, D2, rng)
        tin = tmoments(in)

        for α in (true, false), β in (true, false)
            out = randmoment(T, 1, rng)
            tout0 = tmoments(out)

            H2Factory.toeplitzmul!(out, tf, in, α, β, SerialScheduler())

            expected = (β ? tout0[1] : zero(tout0[1]))
            if α
                for i in 1:D2
                    expected = expected .+ tf[i].toeplitz .* tin[i]
                end
            end
            @test H2Factory.toeplitzmoment(out, 1) ≈ expected

            for i in 1:D2
                @test H2Factory.toeplitzmoment(in, i) == tin[i]
            end
        end
    end

    @testset "toeplitzmul!: vector tf, D2==1 broadcasts one input to all outputs" begin
        # The mirror case: a single input component fanned out through D1 translation operators.
        rng = MersenneTwister(22)
        T = ComplexF64
        D1 = 3
        tf = randtf(T, D1; rng=rng)
        in = randmoment(T, 1, rng)
        tin = tmoments(in)

        for α in (true, false), β in (true, false)
            out = randmoment(T, D1, rng)
            tout0 = tmoments(out)

            H2Factory.toeplitzmul!(out, tf, in, α, β, SerialScheduler())

            for i in 1:D1
                expected = (β ? tout0[i] : zero(tout0[i]))
                α && (expected = expected .+ tf[i].toeplitz .* tin[1])
                @test H2Factory.toeplitzmoment(out, i) ≈ expected
            end
            @test H2Factory.toeplitzmoment(in, 1) == tin[1]
        end
    end

    @testset "toeplitzmul!: scalar tf applies the same operator componentwise" begin
        rng = MersenneTwister(23)
        T = ComplexF64
        D = 3
        tf = H2Factory.ToeplitzMatrixRepresentation(
            randn(rng, T, H2Factory.standardtoeplitzsize(MSIZE))
        )
        in = randmoment(T, D, rng)
        tin = tmoments(in)

        for α in (true, false), β in (true, false)
            out = randmoment(T, D, rng)
            tout0 = tmoments(out)

            H2Factory.toeplitzmul!(out, tf, in, α, β, SerialScheduler())

            for i in 1:D
                expected = (β ? tout0[i] : zero(tout0[i]))
                α && (expected = expected .+ tf.toeplitz .* tin[i])
                @test H2Factory.toeplitzmoment(out, i) ≈ expected
            end
            for i in 1:D
                @test H2Factory.toeplitzmoment(in, i) == tin[i]
            end
        end
    end

    @testset "toeplitzmul!: real element type" begin
        # The FFT domain is complex in production, but nothing in `toeplitzmul!` is complex-specific --
        # a real element type must work identically. Guards against an accidental complex assumption.
        rng = MersenneTwister(24)
        T = Float64
        D = 2
        tf = randtf(T, D, D; rng=rng)
        in = randmoment(T, D, rng)
        tin = tmoments(in)
        out = randmoment(T, D, rng)

        H2Factory.toeplitzmul!(out, tf, in, true, false, SerialScheduler())

        for i in 1:D
            expected = sum(tf[i, j].toeplitz .* tin[j] for j in 1:D)
            @test H2Factory.toeplitzmoment(out, i) ≈ expected
        end
    end
end # @testset "toeplitzmul!"

end # module TestToeplitzMul
