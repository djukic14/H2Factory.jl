module TestSparseAssemblerCompatibility

using Test
using SparseArrays
using BEAST, CompScienceMeshes, H2Factory, H2Trees

@testset "Sparse assembler with registered BEAST" begin
    testmesh = meshsphere(0.5, 0.5)
    trialmesh = meshsphere(0.5, 0.35)
    X = raviartthomas(testmesh)
    Y = raviartthomas(trialmesh)
    @test numfunctions(X) != numfunctions(Y)
    operator = Maxwell3D.singlelayer(; wavenumber=2π)
    quadrature = BEAST.DoubleNumWiltonSauterQStrat(1, 1, 1, 1, 1, 1, 1, 1)
    builder = TwoNTreeBuilder(; minhalfsize=0.125, minvalues=0)
    allnear = (args...) -> true

    for trial in (X, Y)
        tree = if trial === X
            buildtree(X; builder=builder)
        else
            buildtree(X, trial; builder=BlockTreeBuilder(; test=builder, trial=builder))
        end
        assembler = SparseBEASTNearInteractionsAssembler{ComplexF64}(
            operator, X, trial, quadrature
        )
        reference = BEAST.assemble(operator, X, trial; quadstrat=quadrature)
        actual = assembler(tree, allnear; verbose=false)
        @test size(actual) == size(reference)
        @test nnz(actual) > 0
        @test Matrix(actual) ≈ reference rtol = 1e-12
    end
end

end # module TestSparseAssemblerCompatibility
