"""
    RandomEntries()

Matrix data selecting a [`RandomKernelMatrix`](@ref) in [`buildkernelmatrix`](@ref). Use it
as `matrixdata` of an [`AssemblyStrategy`](@ref) to assemble a near field with the sparsity
of the real one but random entries, e.g. to check or measure storage and multiplication
without evaluating the kernel.
"""
struct RandomEntries end

"""
    RandomKernelMatrix{T}(nrows, ncolumns)

Kernel matrix whose entries are independent random values `rand(T)`. Unlike a
[`DummyKernelMatrix`](@ref) it has the near blocks of the tree, so assembling its near
interactions walks the tree and gives the sparsity of the real near field.

Every evaluation draws new values; the entries are not reproducible between calls. The
values come from the task-local default generator, so blocks may be evaluated concurrently.
"""
struct RandomKernelMatrix{T} <: AbstractKernelMatrix{T}
    nrows::Int
    ncolumns::Int

    function RandomKernelMatrix{T}(nrows::Integer, ncolumns::Integer) where {T}
        return new{T}(nrows, ncolumns)
    end
end

Base.size(matrix::RandomKernelMatrix) = (matrix.nrows, matrix.ncolumns)

function (::RandomKernelMatrix{T})(block, rows, columns) where {T}
    for n in 1:length(columns), m in 1:length(rows)
        block[m, n] += rand(T)
    end
    return nothing
end
