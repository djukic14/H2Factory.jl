module TestH2BlockSparseSmoke

using Test
using LinearAlgebra
using BEAST, CompScienceMeshes, BlockSparseMatrices
using H2Factory
using H2Trees
using OhMyThreads: SerialScheduler, DynamicScheduler

# `BlockSparseStorage` builds its matrix in the H2BlockSparse extension, which needs only
# BlockSparseMatrices. Deeper numerical coverage against the sparse storage lives in
# `test/nearinteractions/test_nearinteractions.jl`.

@testset "BlockSparseStorage with a point kernel" begin
    testpoints = collect(range(0.0, 1.0; length=6))
    trialpoints = collect(range(0.5, 2.0; length=4))
    kernel = H2Factory.PointKernelMatrix{Float64}(
        (x, y) -> 1 / (1 + abs(x - y)), testpoints, trialpoints
    )
    rows = [[1, 2, 3], [4, 5, 6]]
    columns = [[1, 2], [3, 4]]
    reference = zeros(6, 4)
    for (r, c) in zip(rows, columns)
        reference[r, c] .= [
            1 / (1 + abs(x - y)) for x in testpoints[r], y in trialpoints[c]
        ]
    end

    for storage in (
        H2Factory.BlockSparseStorage(),
        H2Factory.BlockSparseStorage(; scheduler=DynamicScheduler()),
    )
        A = H2Factory.assembleblocks(kernel, rows, columns, storage, DynamicScheduler())
        @test A isa BlockSparseMatrix
        @test size(A) == (6, 4)
        x = randn(4)
        @test A * x ≈ reference * x
        y = randn(6)
        @test transpose(A) * y ≈ transpose(reference) * y
    end
end

@testset "BlockSparseStorage without near blocks" begin
    # An empty block list must not reach the coloring of a non-serial scheduler.
    kernel = H2Factory.DummyKernelMatrix{ComplexF64}(5, 3)
    strategy = H2Factory.AssemblyStrategy(
        H2Factory.DummyEntries(),
        H2Factory.BlockSparseStorage(; scheduler=DynamicScheduler()),
    )
    A = H2Factory.assemblenearinteractions(
        kernel, nothing, strategy, H2Factory.InteractionStrategy()
    )
    @test A isa BlockSparseMatrix
    @test size(A) == (5, 3)
    @test A * ones(ComplexF64, 3) == zeros(ComplexF64, 5)
end

@testset "BlockSparseStorage with a BEAST operator" begin
    λ = 1.0
    κ = 2 * π / λ
    X = raviartthomas(meshsphere(0.5 * λ, λ / 3))
    op = Maxwell3D.singlelayer(; wavenumber=κ)
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(1, 1, 1, 1, 1, 1, 1, 1)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 3, minvalues=0))
    interaction = H2Factory.InteractionStrategy()

    blockstrategy = H2Factory.AssemblyStrategy(quadstrat, H2Factory.BlockSparseStorage())
    sparsestrategy = H2Factory.AssemblyStrategy(quadstrat, H2Factory.SparseStorage())
    A_block = H2Factory.assemblenearinteractions(op, X, X, tree, blockstrategy, interaction)
    A_sparse = H2Factory.assemblenearinteractions(
        op, X, X, tree, sparsestrategy, interaction
    )

    @test A_block isa BlockSparseMatrix
    @test size(A_block) == (numfunctions(X), numfunctions(X))
    x = randn(ComplexF64, numfunctions(X))
    @test norm(A_block * x - A_sparse * x) / norm(A_sparse * x) < 1e-13
end

end # module TestH2BlockSparseSmoke
