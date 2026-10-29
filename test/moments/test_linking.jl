module TestLinking

using H2Factory
using LinearAlgebra
using Test
@testset "Linking functions" begin
    @test_throws MethodError H2Factory.conjmul!(1, 1, 1)

    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for D in 1:4
            for matrixsize in [(2, 2), (3, 10)]
                mcol = H2Factory.MomentCollection{D,T}(matrixsize)

                for i in eachindex(mcol)
                    el = 1
                    for j in eachindex(mcol[i])
                        mcol[i][j] = el * j
                        el += 1
                    end
                end

                m = H2Factory.Moment{D,T}((matrixsize[1],))

                y = Array(1:matrixsize[2])

                LinearAlgebra.mul!(m, mcol, y)

                for i in eachindex(m)
                    @test m[i] ≈ mcol[i] * y
                end

                for i in eachindex(m)
                    el = 1
                    for j in eachindex(m[i])
                        m[i][j] = el * j
                        el += 1
                    end
                end
                mcopy = deepcopy(m)
                LinearAlgebra.mul!(m, mcol, y, 2, 3)
                for i in eachindex(m)
                    @test m[i] ≈ 3 * mcopy[i] + 2 * mcol[i] * y
                end

                if T <: Complex
                    y = Array(1:matrixsize[2]) * im + Array(matrixsize[2]:-1:1)

                    H2Factory.conjmul!(m, mcol, y)

                    for i in eachindex(m)
                        @test m[i] ≈ mcol[i] * conj(y)
                    end

                    mcopy = deepcopy(m)

                    H2Factory.conjmul!(m, mcol, y, 2, 3)

                    for i in eachindex(m)
                        @test m[i] ≈ 3 * mcopy[i] + 2 * mcol[i] * conj(y)
                    end
                else
                    y = Array(1:matrixsize[2])

                    H2Factory.conjmul!(m, mcol, y)

                    for i in eachindex(m)
                        @test m[i] ≈ mcol[i] * y
                    end

                    mcopy = deepcopy(m)

                    H2Factory.conjmul!(m, mcol, y, 2, 3)

                    for i in eachindex(m)
                        @test m[i] ≈ 3 * mcopy[i] + 2 * mcol[i] * y
                    end
                end

                result = zeros(T, matrixsize[2])
                result2 = zeros(T, matrixsize[2])
                LinearAlgebra.mul!(result, mcol, m)
                for i in eachindex(m)
                    result2 += transpose(mcol[i]) * m[i]
                end

                @test result ≈ result2

                H2Factory.conjmul!(result, mcol, m)
                @test result ≈ conj.(result2)

                result = T.(1:matrixsize[2])

                LinearAlgebra.mul!(result, mcol, m, 2, 3)

                result2 = 3 * T.(1:matrixsize[2])

                for i in eachindex(m)
                    result2 += 2 * transpose(mcol[i]) * m[i]
                end
                @test result ≈ result2

                if D == 1
                    result = T.(1:matrixsize[2])
                    LinearAlgebra.mul!(result, mcol, m, [2], 3)
                    @test result ≈ result2

                    result = T.(1:matrixsize[2])
                    LinearAlgebra.mul!(result, mcol, m, 2, [3])
                    @test result ≈ result2

                    result = T.(1:matrixsize[2])
                    LinearAlgebra.mul!(result, mcol, m, [2], [3])
                    @test result ≈ result2
                end
                α = ones(T, D) * 2
                result = T.(1:matrixsize[2])
                LinearAlgebra.mul!(result, mcol, m, α, 3)
                @test result ≈ result2

                result = T.(1:matrixsize[2])
                H2Factory.conjmul!(result, mcol, m, α, 3)
                @test result ≈ conj.(result2)

                result = T.(1:matrixsize[2])
                H2Factory.conjmul!(result, mcol, m, α[1], 3)
                @test result ≈ conj.(result2)
            end
        end
    end
end

module TestLinkingCreationModule

    using H2Factory
    using Test
    using LinearAlgebra

    @createlinked TestMoment TestMomentCollection
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        for D in 1:4
            for matrixsize in [(2, 2), (3, 10)]
                mcol = TestMomentCollection{D,T}(matrixsize)

                for i in eachindex(mcol)
                    el = 1
                    for j in eachindex(mcol[i])
                        mcol[i][j] = el * j
                        el += 1
                    end
                end

                m = TestMoment{D,T}((matrixsize[1],))

                y = Array(1:matrixsize[2])

                LinearAlgebra.mul!(m, mcol, y)

                for i in eachindex(m)
                    @test m[i] ≈ mcol[i] * y
                end

                for i in eachindex(m)
                    el = 1
                    for j in eachindex(m[i])
                        m[i][j] = el * j
                        el += 1
                    end
                end
                mcopy = deepcopy(m)
                LinearAlgebra.mul!(m, mcol, y, 2, 3)
                for i in eachindex(m)
                    @test m[i] ≈ 3 * mcopy[i] + 2 * mcol[i] * y
                end

                if T <: Complex
                    y = Array(1:matrixsize[2]) * im + Array(matrixsize[2]:-1:1)

                    H2Factory.conjmul!(m, mcol, y)

                    for i in eachindex(m)
                        @test m[i] ≈ mcol[i] * conj(y)
                    end

                    mcopy = deepcopy(m)

                    H2Factory.conjmul!(m, mcol, y, 2, 3)

                    for i in eachindex(m)
                        @test m[i] ≈ 3 * mcopy[i] + 2 * mcol[i] * conj(y)
                    end
                else
                    y = Array(1:matrixsize[2])

                    H2Factory.conjmul!(m, mcol, y)

                    for i in eachindex(m)
                        @test m[i] ≈ mcol[i] * y
                    end

                    mcopy = deepcopy(m)

                    H2Factory.conjmul!(m, mcol, y, 2, 3)

                    for i in eachindex(m)
                        @test m[i] ≈ 3 * mcopy[i] + 2 * mcol[i] * y
                    end
                end

                result = zeros(T, matrixsize[2])
                result2 = zeros(T, matrixsize[2])
                LinearAlgebra.mul!(result, mcol, m)
                for i in eachindex(m)
                    result2 += transpose(mcol[i]) * m[i]
                end

                @test result ≈ result2

                result = T.(1:matrixsize[2])

                LinearAlgebra.mul!(result, mcol, m, 2, 3)

                result2 = 3 * T.(1:matrixsize[2])

                for i in eachindex(m)
                    result2 += 2 * transpose(mcol[i]) * m[i]
                end
                @test result ≈ result2
            end
        end
    end
end

end # module TestLinking
