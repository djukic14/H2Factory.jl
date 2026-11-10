module TestKernelMatrices

using Test
using SparseArrays
using StaticArrays
using LinearAlgebra: norm
using H2Factory
using OhMyThreads: SerialScheduler, StaticScheduler, DynamicScheduler

# Kernel matrices and block assembly need no mesh, no quadrature and no BEAST: a point
# kernel and hand-made blocks pin the storage and scheduling against a dense reference.

struct InverseDistance end
(::InverseDistance)(x, y) = 1 / (1 + abs(x - y))

const TESTPOINTS = collect(range(0.0, 1.0; length=7))
const TRIALPOINTS = collect(range(0.5, 2.0; length=5))
const DENSE = [InverseDistance()(x, y) for x in TESTPOINTS, y in TRIALPOINTS]

# Disjoint blocks covering part of the matrix, including a single row and a single column.
const ROWS = [[1, 2, 3], [4], [5, 6, 7], [2, 3]]
const COLUMNS = [[1, 2], [3, 4, 5], [1], [5]]

function blockreference()
    reference = zeros(size(DENSE))
    for (rows, columns) in zip(ROWS, COLUMNS)
        reference[rows, columns] .= DENSE[rows, columns]
    end
    return reference
end

function pointkernel()
    return H2Factory.PointKernelMatrix{Float64}(InverseDistance(), TESTPOINTS, TRIALPOINTS)
end

@testset "PointKernelMatrix" begin
    kernel = pointkernel()

    @test kernel isa H2Factory.AbstractKernelMatrix{Float64}
    @test size(kernel) == (7, 5)
    @test size(kernel, 1) == 7
    @test eltype(kernel) == Float64

    block = zeros(3, 2)
    kernel(block, [1, 2, 3], [4, 5])
    @test block == DENSE[1:3, 4:5]

    # Blocks are added to, not overwritten.
    kernel(block, [1, 2, 3], [4, 5])
    @test block == 2 * DENSE[1:3, 4:5]

    # Single indices behave like one-element collections.
    entry = zeros(1, 1)
    kernel(entry, 6, 2)
    @test entry[1, 1] == DENSE[6, 2]
end

# A point operator: callable on two points, with its scalar type as `eltype`.
struct PointDistance end
(::PointDistance)(x, y) = ComplexF64(1 / (1 + norm(x - y)))
Base.eltype(::PointDistance) = ComplexF64

@testset "buildkernelmatrix for point collections" begin
    testpoints = [SVector(x, 0.0, 0.0) for x in TESTPOINTS]
    trialpoints = [SVector(0.0, y, 1.0) for y in TRIALPOINTS]
    operator = PointDistance()

    kernel = H2Factory.buildkernelmatrix(operator, testpoints, trialpoints, nothing)
    @test kernel isa H2Factory.PointKernelMatrix{ComplexF64}
    @test size(kernel) == (length(testpoints), length(trialpoints))

    block = zeros(ComplexF64, 2, 3)
    kernel(block, [4, 1], [5, 2, 3])
    @test block == [operator(testpoints[i], trialpoints[j]) for i in [4, 1], j in [5, 2, 3]]

    dummy = H2Factory.buildkernelmatrix(
        operator, testpoints, trialpoints, H2Factory.DummyEntries()
    )
    @test dummy isa H2Factory.DummyKernelMatrix{ComplexF64}
    @test size(dummy) == size(kernel)

    # Point kernels take no matrix data: anything else is not silently ignored.
    @test_throws ErrorException H2Factory.buildkernelmatrix(
        operator, testpoints, trialpoints, :quadstrat
    )
    @test H2Factory.defaultmatrixdata(operator, testpoints, trialpoints) === nothing

    # A functor states its scalar type: no warning.
    @test_logs H2Factory.buildkernelmatrix(operator, testpoints, trialpoints, nothing)

    # A plain function takes the type of its first entry, with a warning.
    distance(x, y) = norm(x - y)
    kernel = @test_logs (:warn,) H2Factory.buildkernelmatrix(
        distance, testpoints, trialpoints, nothing
    )
    @test kernel isa H2Factory.PointKernelMatrix{Float64}
    block = zeros(2, 2)
    kernel(block, [1, 3], [2, 5])
    @test block == [distance(testpoints[i], trialpoints[j]) for i in [1, 3], j in [2, 5]]

    dummy = @test_logs (:warn,) H2Factory.buildkernelmatrix(
        distance, testpoints, trialpoints, H2Factory.DummyEntries()
    )
    @test dummy isa H2Factory.DummyKernelMatrix{Float64}

    random = H2Factory.buildkernelmatrix(
        operator, testpoints, trialpoints, H2Factory.RandomEntries()
    )
    @test random isa H2Factory.RandomKernelMatrix{ComplexF64}
    @test size(random) == size(kernel)
end

@testset "RandomKernelMatrix" begin
    kernel = H2Factory.RandomKernelMatrix{ComplexF32}(7, 5)
    @test kernel isa H2Factory.AbstractKernelMatrix{ComplexF32}
    @test size(kernel) == (7, 5)
    @test eltype(kernel) == ComplexF32

    block = zeros(ComplexF32, 3, 2)
    kernel(block, [1, 4, 6], [2, 5])
    @test all(x -> 0 <= real(x) < 1 && 0 <= imag(x) < 1, block)

    # Single indices behave like one-element collections.
    entry = zeros(ComplexF32, 1, 1)
    kernel(entry, 6, 2)
    @test 0 <= real(entry[1, 1]) < 1

    # The blocks are evaluated: the stored pattern is that of the requested blocks.
    A = H2Factory.assembleblocks(
        H2Factory.RandomKernelMatrix{Float64}(7, 5),
        ROWS,
        COLUMNS,
        H2Factory.SparseStorage(),
        DynamicScheduler(),
    )
    @test A isa SparseMatrixCSC{Float64,Int}
    @test nnz(A) == sum(length.(ROWS) .* length.(COLUMNS))
    @test (A .!= 0) == (blockreference() .!= 0)
end

@testset "SparseStorage with $(nameof(typeof(scheduler)))" for scheduler in (
    SerialScheduler(), StaticScheduler(), DynamicScheduler()
)
    kernel = pointkernel()
    storage = H2Factory.SparseStorage()
    A = H2Factory.assembleblocks(kernel, ROWS, COLUMNS, storage, scheduler)

    @test A isa SparseMatrixCSC{Float64,Int}
    @test size(A) == size(kernel)
    @test nnz(A) == sum(length.(ROWS) .* length.(COLUMNS))
    # Disjoint blocks: every stored entry is written once, so the result is exact.
    @test A == blockreference()
end

@testset "SparseStorage reports progress" begin
    A = H2Factory.assembleblocks(
        pointkernel(),
        ROWS,
        COLUMNS,
        H2Factory.SparseStorage(),
        SerialScheduler();
        verbose=true,
    )
    @test A == blockreference()
end

@testset "SparseStorage keeps the kernel's scalar type" begin
    kernel = H2Factory.PointKernelMatrix{ComplexF32}(
        (x, y) -> ComplexF32(x - y), TESTPOINTS, TRIALPOINTS
    )
    A = H2Factory.assembleblocks(
        kernel, ROWS, COLUMNS, H2Factory.SparseStorage(), SerialScheduler()
    )
    @test eltype(A) == ComplexF32
end

# A kernel matrix that implements only `size`.
struct SizeOnlyKernelMatrix <: H2Factory.AbstractKernelMatrix{Float64} end
Base.size(::SizeOnlyKernelMatrix) = (2, 2)

@testset "AbstractKernelMatrix protocol" begin
    @test eltype(H2Factory.PointKernelMatrix{ComplexF32}) == ComplexF32

    # A missing block method names the protocol instead of failing silently.
    @test_throws ErrorException SizeOnlyKernelMatrix()(zeros(1, 1), 1, 1)

    # No permissive default for unknown kernel families.
    @test_throws ErrorException H2Factory.buildkernelmatrix(
        :operator, :testspace, :trialspace, :matrixdata
    )

    kernel = pointkernel()
    @test H2Factory.localkernelmatrix(kernel, [1, 2], [3]) === kernel
end

@testset "AssemblyStrategy" begin
    strategy = H2Factory.AssemblyStrategy(:matrixdata, H2Factory.SparseStorage())
    @test strategy isa H2Factory.AbstractAssemblyStrategy
    @test H2Factory.matrixdata(strategy) === :matrixdata
    @test H2Factory.storage(strategy) === H2Factory.SparseStorage()
    @test H2Factory.scheduler(strategy) isa DynamicScheduler

    strategy = H2Factory.AssemblyStrategy(
        :matrixdata, H2Factory.SparseStorage(); scheduler=SerialScheduler()
    )
    @test H2Factory.scheduler(strategy) isa SerialScheduler
end

# Deliberately different field names from `AssemblyStrategy`'s, so the default accessors
# cannot satisfy it: a direct field read would fail here.
struct RenamedFieldAssemblyStrategy{D,S,C} <: H2Factory.AbstractAssemblyStrategy
    data::D
    store::S
    tasks::C
end
H2Factory.matrixdata(s::RenamedFieldAssemblyStrategy) = s.data
H2Factory.storage(s::RenamedFieldAssemblyStrategy) = s.store
H2Factory.scheduler(s::RenamedFieldAssemblyStrategy) = s.tasks

@testset "AbstractAssemblyStrategy: assembly reads the accessors" begin
    kernel = H2Factory.DummyKernelMatrix{Float64}(3, 2)
    strategy = RenamedFieldAssemblyStrategy(
        H2Factory.DummyEntries(), H2Factory.SparseStorage(), SerialScheduler()
    )
    A = H2Factory.assemblenearinteractions(
        kernel, nothing, strategy, H2Factory.InteractionStrategy()
    )
    @test A == spzeros(3, 2)
end

@testset "DummyKernelMatrix walks no tree" begin
    kernel = H2Factory.DummyKernelMatrix{ComplexF64}(4, 3)
    @test size(kernel) == (4, 3)
    @test eltype(kernel) == ComplexF64

    # `nothing` as the tree: any traversal would fail.
    rows, columns = H2Factory.nearblocks(kernel, nothing, H2Factory.InteractionStrategy())
    @test isempty(rows) && isempty(columns)

    A = H2Factory.assemblenearinteractions(
        kernel,
        nothing,
        H2Factory.AssemblyStrategy(H2Factory.DummyEntries(), H2Factory.SparseStorage()),
        H2Factory.InteractionStrategy(),
    )
    @test A isa SparseMatrixCSC{ComplexF64,Int}
    @test A == spzeros(ComplexF64, 4, 3)
end

end # module TestKernelMatrices
