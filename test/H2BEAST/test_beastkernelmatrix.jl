module TestBEASTKernelMatrix

using Test
using SparseArrays
using BEAST, CompScienceMeshes, H2Factory, H2Trees

@testset "BEASTKernelMatrix against BEAST.assemble" begin
    X = raviartthomas(meshsphere(0.5, 0.5))
    Y = raviartthomas(meshsphere(0.5, 0.35))
    @test numfunctions(X) != numfunctions(Y)
    operator = Maxwell3D.singlelayer(; wavenumber=2π)
    quadrature = BEAST.DoubleNumWiltonSauterQStrat(1, 1, 1, 1, 1, 1, 1, 1)

    for trial in (X, Y)
        kernel = H2Factory.buildkernelmatrix(operator, X, trial, quadrature)
        @test kernel isa H2Factory.BEASTKernelMatrix{ComplexF64}
        @test size(kernel) == (numfunctions(X), numfunctions(trial))

        reference = BEAST.assemble(operator, X, trial; quadstrat=quadrature)
        block = zeros(ComplexF64, size(kernel))
        kernel(block, 1:numfunctions(X), 1:numfunctions(trial))
        @test block ≈ reference rtol = 1e-12

        # A single entry and a scattered block use BEAST's block-local indices.
        entry = zeros(ComplexF64, 1, 1)
        kernel(entry, 3, 2)
        @test entry[1, 1] ≈ reference[3, 2] rtol = 1e-12

        rows, columns = [5, 1, 4], [2, 6]
        block = zeros(ComplexF64, length(rows), length(columns))
        kernel(block, rows, columns)
        @test block ≈ reference[rows, columns] rtol = 1e-12
    end
end

@testset "BEASTKernelMatrix constructors" begin
    X = raviartthomas(meshsphere(0.5, 0.5))
    operator = Maxwell3D.singlelayer(; wavenumber=2π)

    # The default quadrature and scalar type are BEAST's.
    kernel = H2Factory.BEASTKernelMatrix(operator, X, X)
    @test kernel isa H2Factory.BEASTKernelMatrix{ComplexF64}
    @test size(kernel) == (numfunctions(X), numfunctions(X))

    reference = BEAST.assemble(operator, X, X)
    block = zeros(ComplexF64, 2, 3)
    kernel(block, [1, 2], [3, 4, 5])
    @test block ≈ reference[[1, 2], [3, 4, 5]] rtol = 1e-12

    # An explicit scalar type sets the stored type.
    kernel = H2Factory.BEASTKernelMatrix{ComplexF32}(operator, X, X)
    @test eltype(kernel) == ComplexF32

    # The default matrix data is BEAST's default quadrature.
    @test H2Factory.defaultmatrixdata(operator, X, X) ==
        BEAST.defaultquadstrat(operator, X, X)
end

@testset "Sparse near interactions through the operator" begin
    X = raviartthomas(meshsphere(0.5, 0.5))
    Y = raviartthomas(meshsphere(0.5, 0.35))
    operator = Maxwell3D.singlelayer(; wavenumber=2π)
    quadrature = BEAST.DoubleNumWiltonSauterQStrat(1, 1, 1, 1, 1, 1, 1, 1)
    builder = TwoNTreeBuilder(; minhalfsize=0.125, minvalues=0)
    interaction = H2Factory.InteractionStrategy()
    strategy = H2Factory.AssemblyStrategy(quadrature, H2Factory.SparseStorage())

    for trial in (X, Y)
        tree = if trial === X
            buildtree(X; builder=builder)
        else
            buildtree(X, trial; builder=BlockTreeBuilder(; test=builder, trial=builder))
        end
        actual = H2Factory.assemblenearinteractions(
            operator, X, trial, tree, strategy, interaction
        )
        reference = H2Factory.nearinteractions(
            BEAST.assemble(operator, X, trial; quadstrat=quadrature),
            tree,
            H2Factory.nearcriterion(interaction)(tree),
        )
        @test actual isa SparseMatrixCSC{ComplexF64,Int}
        @test size(actual) == size(reference)
        @test nnz(actual) > 0
        @test actual ≈ reference rtol = 1e-12

        # Random entries walk the same tree: the pattern is that of the real near field.
        randomstrategy = H2Factory.AssemblyStrategy(
            H2Factory.RandomEntries(), H2Factory.SparseStorage()
        )
        random = H2Factory.assemblenearinteractions(
            operator, X, trial, tree, randomstrategy, interaction
        )
        @test random isa SparseMatrixCSC{ComplexF64,Int}
        @test findnz(random)[1:2] == findnz(actual)[1:2]
    end
end

@testset "Dummy near interactions through the operator" begin
    X = raviartthomas(meshsphere(0.5, 0.5))
    operator = Maxwell3D.singlelayer(; wavenumber=2π)
    strategy = H2Factory.AssemblyStrategy(
        H2Factory.DummyEntries(), H2Factory.SparseStorage()
    )

    kernel = H2Factory.buildkernelmatrix(operator, X, X, H2Factory.DummyEntries())
    @test kernel isa H2Factory.DummyKernelMatrix{ComplexF64}

    # `nothing` as the tree: the dummy path walks none.
    A = H2Factory.assemblenearinteractions(
        operator, X, X, nothing, strategy, H2Factory.InteractionStrategy()
    )
    @test A == spzeros(ComplexF64, numfunctions(X), numfunctions(X))
end

# The entries of `kernel[rows, columns]`, evaluated through the block path.
function blockentries(kernel, rows, columns)
    block = zeros(eltype(kernel), length(rows), length(columns))
    kernel(block, rows, columns)
    return block
end

# Rows, columns, blocks and single entries of `localmatrix` against the block path of
# `kernel`. Indices outside the prepared sets exercise the fallback to `kernel`.
function testlocalsampling(localmatrix, kernel, rows, columns, otherrows, othercolumns)
    # Both paths sum the same products in a different order.
    tolerance = 1e4 * eps(real(eltype(kernel)))
    reference = blockentries(kernel, rows, columns)
    for (i, row) in enumerate(rows)
        block = zeros(eltype(kernel), 1, length(columns))
        localmatrix(block, row, columns)
        @test block ≈ reference[i:i, :] rtol = tolerance
    end
    for (j, column) in enumerate(columns)
        block = zeros(eltype(kernel), length(rows), 1)
        localmatrix(block, rows, column)
        @test block ≈ reference[:, j:j] rtol = tolerance
    end
    @test blockentries(localmatrix, rows, columns) ≈ reference rtol = tolerance
    entry = zeros(eltype(kernel), 1, 1)
    localmatrix(entry, rows[2], columns[3])
    @test entry[1, 1] ≈ reference[2, 3] rtol = tolerance

    # A row of an unprepared index against the prepared columns, and vice versa.
    reference = blockentries(kernel, otherrows, columns)
    block = zeros(eltype(kernel), 1, length(columns))
    localmatrix(block, otherrows[1], columns)
    @test block ≈ reference[1:1, :] rtol = tolerance
    reference = blockentries(kernel, rows, othercolumns)
    block = zeros(eltype(kernel), length(rows), 1)
    localmatrix(block, rows, othercolumns[1])
    @test block ≈ reference[:, 1:1] rtol = tolerance
    return nothing
end

@testset "localkernelmatrix for BEAST kernels" begin
    X = raviartthomas(meshsphere(0.5, 0.5))
    Y = raviartthomas(meshsphere(0.5, 0.35))
    operator = Maxwell3D.singlelayer(; wavenumber=2π)
    farquadrature = BEAST.DoubleNumQStrat(2, 3)
    extension = Base.get_extension(H2Factory, :H2BEAST)

    for trial in (X, Y)
        kernel = H2Factory.buildkernelmatrix(operator, X, trial, farquadrature)
        m, n = numfunctions(X), numfunctions(trial)
        rows, columns = [5, 1, 3], [2, 7, 4, 9]
        otherrows, othercolumns = [m, m - 1], [n - 2, n]

        localmatrix = H2Factory.localkernelmatrix(kernel, rows, columns)
        @test localmatrix isa extension.LocalBEASTKernelMatrix
        @test localmatrix isa H2Factory.AbstractKernelMatrix{ComplexF64}
        @test size(localmatrix) == size(kernel)
        testlocalsampling(localmatrix, kernel, rows, columns, otherrows, othercolumns)

        # One-sided preparation: the unprepared side uses the block path.
        for (localrows, localcolumns) in ((rows, nothing), (nothing, columns))
            localmatrix = H2Factory.localkernelmatrix(kernel, localrows, localcolumns)
            @test localmatrix isa extension.LocalBEASTKernelMatrix
            @test (localmatrix.rowdata === nothing) == (localcolumns === nothing)
            @test (localmatrix.columndata === nothing) == (localrows === nothing)
            testlocalsampling(localmatrix, kernel, rows, columns, otherrows, othercolumns)
        end
    end

    # An explicit scalar type sets the stored type; BEAST integrates in its own.
    kernel = H2Factory.BEASTKernelMatrix{ComplexF32}(operator, X, X, farquadrature)
    localmatrix = H2Factory.localkernelmatrix(kernel, [1, 2], [3, 4, 5])
    @test eltype(localmatrix) == ComplexF32
    testlocalsampling(localmatrix, kernel, [1, 2], [3, 4, 5], [6], [7])

    # Singular quadrature keeps the block path: the kernel itself is returned.
    kernel = H2Factory.buildkernelmatrix(
        operator, X, X, BEAST.defaultquadstrat(operator, X, X)
    )
    @test H2Factory.localkernelmatrix(kernel, [1, 2], [3, 4]) === kernel
end

@testset "localkernelmatrix for BEAST kernels (2D)" begin
    line = meshsegment(1.0, 0.05)
    testspace = lagrangec0d1(line)
    trialspace = lagrangec0d1(translate(line, [0.0, 2.0]))
    farquadrature = BEAST.DoubleNumQStrat(3, 4)
    extension = Base.get_extension(H2Factory, :H2BEAST)
    rows, columns = collect(2:6), collect(8:15)

    for operator in (
        Helmholtz2D.singlelayer(; wavenumber=2π),
        Helmholtz2D.doublelayer(; wavenumber=2π),
        Helmholtz2D.hypersingular(; wavenumber=2π),
    )
        kernel = H2Factory.buildkernelmatrix(operator, testspace, trialspace, farquadrature)
        localmatrix = H2Factory.localkernelmatrix(kernel, rows, columns)
        @test localmatrix isa extension.LocalBEASTKernelMatrix
        testlocalsampling(localmatrix, kernel, rows, columns, [1, 7], [1, 16])
    end
end

end # module TestBEASTKernelMatrix
