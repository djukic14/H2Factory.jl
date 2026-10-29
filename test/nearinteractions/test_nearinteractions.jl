module TestNearInteractions

using Test
using SparseArrays, LinearAlgebra
using StaticArrays
using CompScienceMeshes, BEAST
using H2Factory
using H2FactoryTestUtils
using BlockSparseMatrices
using H2Trees

@testset "Near-interactions Galerkin" begin
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
                    A = assemble(op, X, X; quadstrat=quadstrat)

                    assembler = SparseBEASTNearInteractionsAssembler{ComplexF64}(
                        op, X, X, quadstrat
                    )

                    blockassembler = BlockBEASTNearInteractionsAssembler{ComplexF64}(
                        op, X, X, quadstrat
                    )
                    A_sparse = assembler(tree, H2Trees.isnear; verbose=true)

                    A_block = blockassembler(
                        tree, H2Trees.isnear; verbose=true, supportsgalerkinsymmetric=false
                    )

                    A_near = nearinteractions(A, tree, H2Trees.isnear)

                    for nearmatrix in [A_sparse, A_block]
                        A_H2 = FakeH2Matrix(nearmatrix, tree, H2Trees.isnear)

                        @test H2Factory.nearinteractions(A_H2) == nearmatrix
                        @test H2Factory.nearinteractions(transpose(A_H2)) ==
                            transpose(nearmatrix)
                        @test H2Factory.nearinteractions(adjoint(A_H2)) ==
                            adjoint(nearmatrix)
                        @test H2Factory.nearinteractions(A, A_H2) == A_near

                        x = randn(ComplexF64, size(A, 2))
                        @test H2Factory.nearmul(A_H2, x) ≈ nearmatrix * x
                        @test H2Factory.nearmul(transpose(A_H2), x) ≈
                            transpose(nearmatrix) * x
                        @test H2Factory.nearmul(adjoint(A_H2), x) ≈ adjoint(nearmatrix) * x

                        y = randn(ComplexF64, size(A, 1))
                        @test H2Factory.nearmul!(y, A_H2, x) ≈ nearmatrix * x
                        @test H2Factory.nearmul!(y, transpose(A_H2), x) ≈
                            transpose(nearmatrix) * x
                        @test H2Factory.nearmul!(y, adjoint(A_H2), x) ≈
                            adjoint(nearmatrix) * x

                        y2 = deepcopy(y)
                        α = randn(ComplexF64)
                        β = randn(ComplexF64)
                        @test H2Factory.nearmul!(y, A_H2, x, α, β) ≈
                            LinearAlgebra.mul!(y2, nearmatrix, x, α, β)
                    end

                    println("Compression: ", nnz(A_sparse) / prod(size(A)))

                    error = maximum(norm.(A_near - A_sparse) / maximum(norm.(A_near)))

                    @show error
                    @test error < 1e-14

                    randomassembler = RandomNearInteractionsAssembler{ComplexF64}(
                        op, X, X, quadstrat
                    )

                    A_random = randomassembler(tree, H2Trees.isnear)

                    @test nnz(A_random) == nnz(A_sparse)

                    dummyassembler = DummyNearInteractionsAssembler{ComplexF64}(
                        op, X, X, quadstrat
                    )

                    A_dummy = dummyassembler(tree, H2Trees.isnear)

                    @test A_dummy == spzeros(ComplexF64, size(A))

                    for i in axes(A_sparse, 2)
                        x = randn(ComplexF64, size(A_sparse, 2))
                        er =
                            maximum.(norm(A_sparse * x - A_block * x)) /
                            maximum(norm.(A_block * x))

                        @test er < 1e-14

                        er =
                            maximum.(norm(adjoint(A_sparse) * x - adjoint(A_block) * x)) /
                            maximum(norm.(adjoint(A_sparse) * x))
                        @test er < 1e-14

                        er =
                            maximum.(
                                norm(transpose(A_sparse) * x - transpose(A_block) * x)
                            ) / maximum(norm.(adjoint(A_sparse) * x))
                        @test er < 1e-14
                    end
                end
            end
        end
    end
end

@testset "Near-interactions Petrov-Galerkin" begin
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

                        A = assemble(op, X, Y; quadstrat=quadstrat)

                        assembler = SparseBEASTNearInteractionsAssembler{ComplexF64}(
                            op, X, Y, quadstrat
                        )
                        blockassembler = BlockBEASTNearInteractionsAssembler{ComplexF64}(
                            op, X, Y, quadstrat
                        )

                        A_sparse = assembler(tree, H2Trees.isnear; verbose=true)
                        A_block = blockassembler(tree, H2Trees.isnear; verbose=true)
                        nnz(A_sparse) == 0 && continue
                        println("Compression: ", nnz(A_sparse) / prod(size(A)))

                        for i in axes(A_sparse, 2)
                            x = randn(ComplexF64, size(A_sparse, 2))
                            er = maximum.(norm(A_sparse * x - A_block * x))
                            @test er < 1e-14
                        end

                        for i in axes(A_sparse, 1)
                            x = randn(ComplexF64, size(A_sparse, 1))

                            er = maximum.(
                                norm(adjoint(A_sparse) * x - adjoint(A_block) * x)
                            )
                            @test er < 1e-14

                            er = maximum.(
                                norm(transpose(A_sparse) * x - transpose(A_block) * x)
                            )
                            @test er < 1e-14
                        end

                        A_near = nearinteractions(A, tree, H2Trees.isnear)

                        error = maximum(norm.(A_near - A_sparse) / maximum(norm.(A_near)))

                        for nearmatrix in [A_sparse, A_block]
                            A_H2 = FakeH2Matrix(nearmatrix, tree, H2Trees.isnear)

                            @test H2Factory.nearinteractions(A_H2) == nearmatrix
                            @test H2Factory.nearinteractions(transpose(A_H2)) ==
                                transpose(nearmatrix)
                            @test H2Factory.nearinteractions(adjoint(A_H2)) ==
                                adjoint(nearmatrix)
                            @test H2Factory.nearinteractions(A, A_H2) == A_near

                            x = randn(ComplexF64, size(A, 2))
                            y = randn(ComplexF64, size(A, 1))

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

                            x = randn(ComplexF64, size(A, 2))
                            y = randn(ComplexF64, size(A, 1))

                            y2 = deepcopy(y)
                            x2 = deepcopy(x)
                            α = randn(ComplexF64)
                            β = randn(ComplexF64)
                            @test H2Factory.nearmul!(y, A_H2, x, α, β) ≈
                                LinearAlgebra.mul!(y2, nearmatrix, x, α, β)
                            @test H2Factory.nearmul!(x, transpose(A_H2), y, α, β) ≈
                                LinearAlgebra.mul!(x2, transpose(nearmatrix), y2, α, β)
                            @test H2Factory.nearmul!(x, A_H2', y, α, β) ≈
                                LinearAlgebra.mul!(x2, nearmatrix', y2, α, β)
                        end

                        @show error
                        @test error < 1e-14

                        randomassembler = RandomNearInteractionsAssembler{ComplexF64}(
                            op, X, Y, quadstrat
                        )

                        A_random = randomassembler(tree, H2Trees.isnear)

                        @test nnz(A_random) == nnz(A_sparse)
                        A_random[.!iszero.(A_random)] .= 1
                        A_sparse[.!iszero.(A_sparse)] .= 1

                        @test A_random == A_sparse

                        dummyassembler = DummyNearInteractionsAssembler{ComplexF64}(
                            op, X, Y, quadstrat
                        )

                        A_dummy = dummyassembler(tree, H2Trees.isnear)

                        @test A_dummy == spzeros(ComplexF64, size(A))
                    end
                end
            end
        end
    end
end

@testset "Galerkinsymmetric Blocknearassembler" begin
    λ = 1.0
    κ = 2 * π / λ

    m1 = meshsphere(0.5 * λ, λ / 5)
    m2 = meshcuboid(0.5 * λ, 0.5 * λ, 0.5 * λ, λ / 5)
    singlelayer = Maxwell3D.singlelayer(; wavenumber=κ)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)

    for op in [singlelayer, doublelayer]
        for m in [m1, m2]
            for testf in [raviartthomas, buffachristiansen]
                X = testf(m)
                quadstrat = BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 7, 7, 7, 7)
                tree = buildtree(
                    X;
                    builder=TwoNTreeBuilder(;
                        minhalfsize=λ / 8,
                        minvalues=0,
                        protrusion=H2Trees.NoProtrusionCheck(),
                    ),
                )

                assembler = SparseBEASTNearInteractionsAssembler{ComplexF64}(
                    op, X, X, quadstrat
                )

                blockassembler = BlockBEASTNearInteractionsAssembler{ComplexF64}(
                    op, X, X, quadstrat
                )
                A_sparse = assembler(
                    tree, H2Trees.isnear; verbose=true, supportsgalerkinsymmetric=false
                )

                A_block = blockassembler(
                    tree, H2Trees.isnear; verbose=true, supportsgalerkinsymmetric=true
                )

                x = zeros(ComplexF64, size(A_sparse, 2))
                for i in 1:size(A_sparse, 2)
                    x .= 0
                    x[i] = 1
                    @test A_sparse * x ≈ A_block * x atol = 1e-4
                end
            end
        end
    end
end

@testset "Abstract Assembler API" begin
    struct Assembler{T} <: H2Factory.AbstractNearInteractionsAssembler{T}
        function Assembler{T}(operator, testspace, trialspace, quadstrat) where {T}
            return new{T}()
        end
    end

    Base.size(::Assembler) = (1, 2)

    assembler = Assembler{ComplexF64}(1, 2, 3, 4)

    # AN ASSEMBLER THAT DOES NOT IMPLEMENT `assemble` MUST FAIL, not succeed quietly.
    #
    # These two were `@test_nowarn` and passed because H2Factory carried empty-bodied `assemble`
    # methods on the abstract type. That made an incomplete assembler silently VALID: it assembled
    # nothing, returned `nothing`, and produced a map missing its entire near-field block while every
    # construction step reported success. The fallbacks are gone now that `assemble` is declared
    # protocol, so the absence is reported. `Assembler` above deliberately implements only `size`.
    @test_throws MethodError H2Factory.assemble(assembler, nothing, nothing, true)
    @test_throws MethodError H2Factory.assemble(assembler, nothing, nothing, nothing, true)

    @test size(assembler, 1) == 1
    @test size(assembler, 2) == 2
end

@testset "an intentional no-op assembler states itself" begin
    # The counterpart to removing the fallbacks: doing nothing is a legitimate answer, but it has to
    # be WRITTEN. These are why nothing depended on the empty abstract methods.

    # The dummy takes its dimensions from the tree and returns before generic assembly starts.
    space = raviartthomas(meshsphere(0.5, 0.5))
    tree = buildtree(space)
    dummy = H2Factory.DummyNearInteractionsAssembler{ComplexF64}(
        nothing, space, space, nothing
    )
    @test dummy(tree, H2Trees.isnear) ==
        spzeros(ComplexF64, H2Trees.numberofvalues(tree), H2Trees.numberofvalues(tree))

    # The pass-through opts out the other way: it returns the matrix it was handed, without
    # traversing or assembling anything. That BEHAVIOUR is the contract pinned here.
    #
    # The current implementation achieves it by short-circuiting before `assemble` ever runs: its
    # two-argument functor is more specific than the abstract assembler's `(tree, isnear)` functor,
    # so it wins dispatch and the traversal never starts. That is why removing the abstract
    # `assemble` fallbacks could not affect it. Deliberately NOT asserted as an absence of methods:
    # a future explicit forwarding method would be harmless, and a test that rejected it would be
    # pinning an implementation detail rather than the contract.
    stored = sparse([1], [1], ComplexF64[2.0], 3, 3)
    passthrough = H2Factory.PassThroughNearInteractionsAssembler{ComplexF64}(
        nothing, nothing, nothing, stored
    )
    @test passthrough(nothing, nothing) === stored

    # And no abstract-typed fallback has crept back: every `assemble` method must dispatch on a
    # CONCRETE assembler, never on the supertype.
    for m in methods(H2Factory.assemble)
        params = Base.unwrap_unionall(m.sig).parameters
        length(params) >= 2 || continue
        @test params[2] !== H2Factory.AbstractNearInteractionsAssembler
    end
end

end # module TestNearInteractions
