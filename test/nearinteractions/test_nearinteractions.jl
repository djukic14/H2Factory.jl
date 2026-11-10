module TestNearInteractions

using Test
using SparseArrays, LinearAlgebra
using StaticArrays
using CompScienceMeshes, BEAST
using H2Factory
using H2FactoryTestUtils
using BlockSparseMatrices
using H2Trees
using H2Factory: AssemblyStrategy, SparseStorage, BlockSparseStorage, DummyEntries
include("../timing.jl")
using .TestTiming: timed

@testset "Near-interactions Galerkin" begin
    rng = H2FactoryTestUtils.checkrng()
    λ = 1.0
    κ = 2 * π / λ
    distance = 5 * λ
    height = λ / 2
    meshsize = λ / 10

    m1 = meshsphere(0.5 * λ, λ / 5)
    m2 = meshcuboid(0.5 * λ, 0.5 * λ, 0.5 * λ, λ / 5)
    singlelayer = Maxwell3D.singlelayer(; wavenumber=κ)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)

    struct FakeH2Matrix{T,N,TREE,ISNEAR} <: H2Factory.GalerkinH2Map{T}
        nearinteractions::N
        tree::TREE
        isnear::ISNEAR
        function FakeH2Matrix(
            nearinteractions::N, tree::TREE, isnear::ISNEAR
        ) where {N,TREE,ISNEAR}
            return new{eltype(N),N,TREE,ISNEAR}(nearinteractions, tree, isnear)
        end
    end

    for op in [singlelayer, doublelayer]
        for m in [m1, m2]
            for testf in [raviartthomas, buffachristiansen]
                X = testf(m)
                quadstrat = BEAST.DoubleNumWiltonSauterQStrat(1, 1, 1, 1, 1, 1, 1, 1)
                for makeit in [
                    X -> buildtree(
                        X; builder=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0)
                    ),
                ]
                    tree = makeit(X)
                    meshname = m === m1 ? "sphere" : "cuboid"
                    label = "Galerkin $(nameof(typeof(op))) $meshname $(nameof(testf)) dofs=$(numfunctions(X)) quadrature=1"
                    A = timed(
                        () -> assemble(op, X, X; quadstrat=quadstrat), "$label reference"
                    )

                    interaction = H2Factory.InteractionStrategy()
                    isnear = H2Factory.nearcriterion(interaction)(tree)
                    A_sparse = timed("$label sparse") do
                        H2Factory.assemblenearinteractions(
                            op,
                            X,
                            X,
                            tree,
                            AssemblyStrategy(quadstrat, SparseStorage()),
                            interaction;
                            verbose=true,
                        )
                    end
                    A_block = timed("$label block") do
                        H2Factory.assemblenearinteractions(
                            op,
                            X,
                            X,
                            tree,
                            AssemblyStrategy(quadstrat, BlockSparseStorage()),
                            interaction;
                            verbose=true,
                        )
                    end

                    A_near = nearinteractions(A, tree, isnear)

                    for nearmatrix in [A_sparse, A_block]
                        A_H2 = FakeH2Matrix(nearmatrix, tree, isnear)

                        @test H2Factory.nearinteractions(A_H2) == nearmatrix
                        @test H2Factory.nearinteractions(transpose(A_H2)) ==
                            transpose(nearmatrix)
                        @test H2Factory.nearinteractions(adjoint(A_H2)) ==
                            adjoint(nearmatrix)
                        @test H2Factory.nearinteractions(A, A_H2) == A_near

                        x = randn(rng, ComplexF64, size(A, 2))
                        @test H2Factory.nearmul(A_H2, x) ≈ nearmatrix * x
                        @test H2Factory.nearmul(transpose(A_H2), x) ≈
                            transpose(nearmatrix) * x
                        @test H2Factory.nearmul(adjoint(A_H2), x) ≈ adjoint(nearmatrix) * x

                        y = randn(rng, ComplexF64, size(A, 1))
                        @test H2Factory.nearmul!(y, A_H2, x) ≈ nearmatrix * x
                        @test H2Factory.nearmul!(y, transpose(A_H2), x) ≈
                            transpose(nearmatrix) * x
                        @test H2Factory.nearmul!(y, adjoint(A_H2), x) ≈
                            adjoint(nearmatrix) * x

                        y2 = deepcopy(y)
                        α = randn(rng, ComplexF64)
                        β = randn(rng, ComplexF64)
                        @test H2Factory.nearmul!(y, A_H2, x, α, β) ≈
                            LinearAlgebra.mul!(y2, nearmatrix, x, α, β)
                    end

                    println("Compression: ", nnz(A_sparse) / prod(size(A)))

                    error = maximum(norm.(A_near - A_sparse) / maximum(norm.(A_near)))

                    @show error
                    @test error < 1e-14

                    A_dummy = H2Factory.assemblenearinteractions(
                        op,
                        X,
                        X,
                        tree,
                        AssemblyStrategy(DummyEntries(), SparseStorage()),
                        interaction,
                    )

                    @test A_dummy == spzeros(ComplexF64, size(A))

                    for i in axes(A_sparse, 2)
                        x = randn(rng, ComplexF64, size(A_sparse, 2))
                        # Sparse and block storage accumulate products in different orders.
                        er = norm(A_sparse * x - A_block * x) / norm(A_block * x)

                        @test er < 1e-13

                        er =
                            norm(adjoint(A_sparse) * x - adjoint(A_block) * x) /
                            norm(adjoint(A_block) * x)
                        @test er < 1e-13

                        er =
                            norm(transpose(A_sparse) * x - transpose(A_block) * x) /
                            norm(transpose(A_block) * x)
                        @test er < 1e-13
                    end
                end
            end
        end
    end
end

@testset "Near-interactions Petrov-Galerkin" begin
    rng = H2FactoryTestUtils.checkrng()
    λ = 1.0
    κ = 2 * π / λ

    m1 = meshsphere(0.5 * λ, λ / 5)
    m2 = translate!(meshsphere(0.6 * λ, λ / 5), SVector(0.5 * λ, 0, 0))
    m3 = translate!(meshsphere(0.6 * λ, λ / 5), SVector(10 * λ, 0, 0))

    singlelayer = Maxwell3D.singlelayer(; wavenumber=κ)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)

    struct FakeH2Matrix{T,N,TREE,ISNEAR} <: H2Factory.GalerkinH2Map{T}
        nearinteractions::N
        tree::TREE
        isnear::ISNEAR
        function FakeH2Matrix(
            nearinteractions::N, tree::TREE, isnear::ISNEAR
        ) where {N,TREE,ISNEAR}
            return new{eltype(N),N,TREE,ISNEAR}(nearinteractions, tree, isnear)
        end
    end

    for op in [singlelayer, doublelayer]
        for my in [m1, m2, m3]
            for trialf in [raviartthomas]
                Y = trialf(my)
                for mx in [m1, m2, m3]
                    for testf in [raviartthomas, buffachristiansen]
                        X = testf(mx)
                        quadstrat = BEAST.DoubleNumWiltonSauterQStrat(
                            1, 1, 1, 1, 1, 1, 1, 1
                        )

                        tree = buildtree(
                            X,
                            Y;
                            builder=BlockTreeBuilder(;
                                test=TwoNTreeBuilder(; minhalfsize=λ / 20, minvalues=0),
                                trial=TwoNTreeBuilder(; minhalfsize=λ / 20, minvalues=0),
                            ),
                        )

                        testmesh = if mx === m1
                            1
                        elseif mx === m2
                            2
                        else
                            3
                        end
                        trialmesh = if my === m1
                            1
                        elseif my === m2
                            2
                        else
                            3
                        end
                        label = "Petrov $(nameof(typeof(op))) meshes=$testmesh,$trialmesh $(nameof(testf))/$(nameof(trialf)) dofs=$(numfunctions(X)),$(numfunctions(Y)) quadrature=1"
                        A = timed(
                            () -> assemble(op, X, Y; quadstrat=quadstrat),
                            "$label reference",
                        )

                        interaction = H2Factory.InteractionStrategy()
                        isnear = H2Factory.nearcriterion(interaction)(tree)
                        A_sparse = timed("$label sparse") do
                            H2Factory.assemblenearinteractions(
                                op,
                                X,
                                Y,
                                tree,
                                AssemblyStrategy(quadstrat, SparseStorage()),
                                interaction;
                                verbose=true,
                            )
                        end
                        A_block = timed("$label block") do
                            H2Factory.assemblenearinteractions(
                                op,
                                X,
                                Y,
                                tree,
                                AssemblyStrategy(quadstrat, BlockSparseStorage()),
                                interaction;
                                verbose=true,
                            )
                        end
                        nnz(A_sparse) == 0 && continue
                        println("Compression: ", nnz(A_sparse) / prod(size(A)))

                        for i in axes(A_sparse, 2)
                            x = randn(rng, ComplexF64, size(A_sparse, 2))
                            er = norm(A_sparse * x - A_block * x) / norm(A_block * x)
                            @test er < 1e-13
                        end

                        for i in axes(A_sparse, 1)
                            x = randn(rng, ComplexF64, size(A_sparse, 1))

                            er =
                                norm(adjoint(A_sparse) * x - adjoint(A_block) * x) /
                                norm(adjoint(A_block) * x)
                            @test er < 1e-13

                            er =
                                norm(transpose(A_sparse) * x - transpose(A_block) * x) /
                                norm(transpose(A_block) * x)
                            @test er < 1e-13
                        end

                        A_near = nearinteractions(A, tree, isnear)

                        error = maximum(norm.(A_near - A_sparse) / maximum(norm.(A_near)))

                        for nearmatrix in [A_sparse, A_block]
                            A_H2 = FakeH2Matrix(nearmatrix, tree, isnear)

                            @test H2Factory.nearinteractions(A_H2) == nearmatrix
                            @test H2Factory.nearinteractions(transpose(A_H2)) ==
                                transpose(nearmatrix)
                            @test H2Factory.nearinteractions(adjoint(A_H2)) ==
                                adjoint(nearmatrix)
                            @test H2Factory.nearinteractions(A, A_H2) == A_near

                            x = randn(rng, ComplexF64, size(A, 2))
                            y = randn(rng, ComplexF64, size(A, 1))

                            @test H2Factory.nearmul(A_H2, x) ≈ nearmatrix * x
                            @test H2Factory.nearmul(transpose(A_H2), y) ≈
                                transpose(nearmatrix) * y
                            @test H2Factory.nearmul(adjoint(A_H2), y) ≈
                                adjoint(nearmatrix) * y

                            @test H2Factory.nearmul!(y, A_H2, x) ≈ nearmatrix * x
                            @test H2Factory.nearmul!(x, transpose(A_H2), y) ≈
                                transpose(nearmatrix) * y
                            @test H2Factory.nearmul!(x, adjoint(A_H2), y) ≈
                                adjoint(nearmatrix) * y

                            x = randn(rng, ComplexF64, size(A, 2))
                            y = randn(rng, ComplexF64, size(A, 1))

                            y2 = deepcopy(y)
                            x2 = deepcopy(x)
                            α = randn(rng, ComplexF64)
                            β = randn(rng, ComplexF64)
                            @test H2Factory.nearmul!(y, A_H2, x, α, β) ≈
                                LinearAlgebra.mul!(y2, nearmatrix, x, α, β)
                            @test H2Factory.nearmul!(x, transpose(A_H2), y, α, β) ≈
                                LinearAlgebra.mul!(x2, transpose(nearmatrix), y2, α, β)
                            @test H2Factory.nearmul!(x, A_H2', y, α, β) ≈
                                LinearAlgebra.mul!(x2, nearmatrix', y2, α, β)
                        end

                        @show error
                        @test error < 1e-14

                        A_dummy = H2Factory.assemblenearinteractions(
                            op,
                            X,
                            Y,
                            tree,
                            AssemblyStrategy(DummyEntries(), SparseStorage()),
                            interaction,
                        )

                        @test A_dummy == spzeros(ComplexF64, size(A))
                    end
                end
            end
        end
    end
end

end # module TestNearInteractions
