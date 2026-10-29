module TestH2BlockSparseSmoke

using Test
using BEAST, CompScienceMeshes, BlockSparseMatrices
using H2Factory
using H2Trees

# `BlockBEASTNearInteractionsAssembler` (the struct) lives in core H2Factory, but its
# `H2Factory.assemble`/`assembleblock`/`blockassembler` methods -- the actual computation this
# extension exists for -- are only defined in `ext/H2BlockSparse/*.jl`, gated on
# `BlockSparseMatrices` being loaded. Deeper correctness coverage of this same code path already
# exists in `test/nearinteractions/test_nearinteractions.jl` (`A_block` compared against a dense
# reference); this is deliberately a much smaller, faster construction/dispatch sanity check, not
# a duplicate of that numerical validation.
@testset "H2BlockSparse smoke" begin
    λ = 1.0
    κ = 2 * π / λ
    m = meshsphere(0.5 * λ, λ / 3)
    X = raviartthomas(m)
    op = Maxwell3D.singlelayer(; wavenumber=κ)
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(1, 1, 1, 1, 1, 1, 1, 1)

    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 3, minvalues=0))

    blockassembler = BlockBEASTNearInteractionsAssembler{ComplexF64}(op, X, X, quadstrat)
    A_block = blockassembler(tree, H2Trees.isnear; verbose=false)

    @test size(A_block) == (numfunctions(X), numfunctions(X))

    x = randn(ComplexF64, numfunctions(X))
    @test_nowarn A_block * x

    # Exercise the adapter against BEAST's full matrix, independent of the near partition.
    extension = Base.get_extension(H2Factory, :H2BlockSparse)
    assembleblock = extension.blockassembler(blockassembler)
    reference = BEAST.assemble(op, X, X; quadstrat=quadstrat)
    assembled = zeros(ComplexF64, size(reference))
    ids = collect(1:numfunctions(X))
    assembleblock(ids, ids, (value, row, col) -> assembled[row, col] += value)
    @test assembled ≈ reference
end

end # module TestH2BlockSparseSmoke
