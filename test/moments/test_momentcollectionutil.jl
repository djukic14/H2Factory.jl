module TestMomentCollectionUtil

using Test
using H2Factory
using LinearAlgebra

@testset "D-Dimensional MomentCollection" begin
    arraysize = [(2, 2), (3, 3)]
    @createmomentcollection FakeMomentCollection
    MomentTypes = [H2Factory.Moment, FakeMomentCollection]

    for MomentType in MomentTypes
        for T in [Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64]
            for arraysize in arraysize
                for D in 1:4
                    m = MomentType{D,Float64}(arraysize)

                    @test H2Factory.dim(m) == D
                    for i in eachindex(H2Factory.moments(m))
                        @test size(H2Factory.moments(m)[i]) == arraysize
                        @test eltype(m) == typeof(H2Factory.moments(m)[i])
                    end

                    for m2 in [MomentType(m), typeof(m)(m)]
                        Base.copyto!(m, m2)
                        for i in eachindex(H2Factory.moments(m))
                            for j in eachindex(H2Factory.moments(m)[i])
                                if isnan(m2.moments[i][j])
                                    @test isnan(m.moments[i][j])
                                    @test isnan(m[i][j])
                                else
                                    @test m.moments[i][j] == m2.moments[i][j]
                                    @test m[i][j] == m.moments[i][j]
                                end
                            end
                        end
                    end

                    m1 = MomentType{D,Float64}(arraysize)
                    for i in eachindex(m)
                        element = 1
                        for j in eachindex(m[i])
                            m[i][j] = element * j
                            element += 1
                        end
                    end

                    m2 = deepcopy(m)

                    LinearAlgebra.mul!(m, m, 2.1)

                    for i in eachindex(m)
                        for j in eachindex(m[i])
                            @test m[i][j] == 2.1 * m2[i][j]
                        end
                    end

                    m = deepcopy(m2)
                    LinearAlgebra.mul!(m, m, 2.1, 0.1, 1.1)

                    for i in eachindex(m)
                        for j in eachindex(m[i])
                            @test m[i][j] ≈ 2.1 * 0.1 * m2[i][j] + 1.1 * m2[i][j]
                        end
                    end
                end
            end
        end

        for T in [Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64]
            m1 = MomentType{2,T}((2, 2))
            m2 = MomentType{2,T}((2, 2))

            collect(m1.moments[i] .= 1 for i in eachindex(m1.moments))
            collect(m2.moments[i] .= 2 for i in eachindex(m2.moments))

            mconj = conj(m1)
            collect(
                @test (mconj.moments[i] == conj(m1.moments[i])) for
                i in eachindex(mconj.moments)
            )

            H2Factory.add!(m1, m2)
            collect(@test all(m1.moments[i] .== 3) for i in eachindex(m1.moments))

            conj!(m1)

            collect(@test all(m1.moments[i] .== conj(T(3))) for i in eachindex(m1.moments))
        end

        for T in [ComplexF16, ComplexF32, ComplexF64]
            m1 = MomentType{2,T}((2, 2))

            collect(m1.moments[i] .= 1 * im for i in eachindex(m1.moments))

            mconj = conj(m1)
            collect(
                @test (mconj.moments[i] == conj(m1.moments[i])) for
                i in eachindex(mconj.moments)
            )

            conj!(m1)

            collect(
                @test all(m1.moments[i] .== conj(T(1) * im)) for i in eachindex(m1.moments)
            )
        end
    end
end

@testset "1,2,3,4D MomentCollection" begin
    for matrixsize in [(1, 2), (100, 10)]
        m1 = H2Factory.MomentCollection1D{Float64}(matrixsize)
        m2 = H2Factory.MomentCollection2D{Float64}(matrixsize)
        m3 = H2Factory.MomentCollection3D{Float64}(matrixsize)
        m4 = H2Factory.MomentCollection4D{Float64}(matrixsize)

        @test H2Factory.dim(m1) == 1
        @test H2Factory.dim(m2) == 2
        @test H2Factory.dim(m3) == 3
        @test H2Factory.dim(m4) == 4

        @test typeof(H2Factory.MomentCollection1D(m1)) == typeof(m1)
        @test typeof(H2Factory.MomentCollection2D(m2)) == typeof(m2)
        @test typeof(H2Factory.MomentCollection3D(m3)) == typeof(m3)
        @test typeof(H2Factory.MomentCollection4D(m4)) == typeof(m4)

        m1_2 = H2Factory.MomentCollection1D(m1)
        m2_2 = H2Factory.MomentCollection2D(m2)
        m3_2 = H2Factory.MomentCollection3D(m3)
        m4_2 = H2Factory.MomentCollection4D(m4)

        m1_3 = typeof(m1)(m1)
        m2_3 = typeof(m2)(m2)
        m3_3 = typeof(m3)(m3)
        m4_3 = typeof(m4)(m4)

        @test H2Factory.dim(m1) == 1
        @test H2Factory.dim(m2) == 2
        @test H2Factory.dim(m3) == 3
        @test H2Factory.dim(m4) == 4

        @test H2Factory.dim(m1_2) == 1
        @test H2Factory.dim(m2_2) == 2
        @test H2Factory.dim(m3_2) == 3
        @test H2Factory.dim(m4_2) == 4

        @test H2Factory.dim(m1_3) == 1
        @test H2Factory.dim(m2_3) == 2
        @test H2Factory.dim(m3_3) == 3
        @test H2Factory.dim(m4_3) == 4

        @test all(size(m1[i]) == size(m1_2[i]) == size(m1_3[i]) for i in eachindex(m1))

        @test all(size(m2[i]) == size(m2_2[i]) == size(m2_3[i]) for i in eachindex(m2))

        @test all(size(m3[i]) == size(m3_2[i]) == size(m3_3[i]) for i in eachindex(m3))
    end

    matrixsize = (10, 100)
    m1 = H2Factory.MomentCollection1D{Float64}(matrixsize)
    m2 = H2Factory.MomentCollection2D{Float64}(matrixsize)
    m3 = H2Factory.MomentCollection3D{Float64}(matrixsize)
    m4 = H2Factory.MomentCollection4D{Float64}(matrixsize)

    for m in [m1, m2, m3, m4]
        for i in eachindex(m)
            el = 1
            for j in eachindex(m[i])
                m[i][j] = el * i
                el += 1
            end
        end
    end

    m1_split = H2Factory.subset(m1, [1, 2, 99])
    for i in eachindex(m1)
        @test m1[i][:, [1, 2, 99]] == m1_split[i]
    end
    @test_throws BoundsError H2Factory.subset(m1, [101, 102])
    @test typeof(m1_split) == typeof(m1)

    m2_split = H2Factory.subset(m2, [1, 2, 99])
    @test_throws BoundsError H2Factory.subset(m2, [101, 102])
    @test typeof(m2_split) == typeof(m2)
    for i in eachindex(m2)
        @test m2[i][:, [1, 2, 99]] == m2_split[i]
    end

    m3_split = H2Factory.subset(m3, [1, 2, 99])
    for i in eachindex(m3)
        @test m3[i][:, [1, 2, 99]] == m3_split[i]
    end
    @test_throws BoundsError H2Factory.subset(m3, [101, 102])

    m4_split = H2Factory.subset(m4, [1, 2, 99])
    for i in eachindex(m4)
        @test m4[i][:, [1, 2, 99]] == m4_split[i]
    end
    @test_throws BoundsError H2Factory.subset(m4, [101, 102])

    coefficients = zeros(matrixsize)
    el = 1
    for i in eachindex(coefficients)
        coefficients[i] = el
        el += 1
    end

    m1_elementwise = H2Factory.MomentCollection1D(m1)
    elementwisemul!(m1_elementwise, m1, coefficients)
    for i in eachindex(m1)
        @test m1_elementwise[i] == m1[i] .* coefficients
    end

    m2_elementwise = H2Factory.MomentCollection2D(m2)
    elementwisemul!(m2_elementwise, m2, coefficients)
    for i in eachindex(m2)
        @test m2_elementwise[i] == m2[i] .* coefficients
    end

    m3_elementwise = H2Factory.MomentCollection3D(m3)
    elementwisemul!(m3_elementwise, m3, coefficients)
    for i in eachindex(m3)
        @test m3_elementwise[i] == m3[i] .* coefficients
    end

    m4_elementwise = H2Factory.MomentCollection4D(m4)
    elementwisemul!(m4_elementwise, m4, coefficients)
    for i in eachindex(m4)
        @test m4_elementwise[i] == m4[i] .* coefficients
    end

    m1_elementwise = deepcopy(m1)
    elementwisemul!(m1, m1, coefficients)
    for i in eachindex(m1)
        @test m1[i] == m1_elementwise[i] .* coefficients
    end

    m2_elementwise = deepcopy(m2)
    elementwisemul!(m2, m2, coefficients)
    for i in eachindex(m2)
        @test m2[i] == m2_elementwise[i] .* coefficients
    end

    m3_elementwise = deepcopy(m3)
    elementwisemul!(m3, m3, coefficients)
    for i in eachindex(m3)
        @test m3[i] == m3_elementwise[i] .* coefficients
    end

    m4_elementwise = deepcopy(m4)
    elementwisemul!(m4, m4, coefficients)
    for i in eachindex(m4)
        @test m4[i] == m4_elementwise[i] .* coefficients
    end
end

end # module TestMomentCollectionUtil
