"""
    DummyEntries()

Matrix data selecting a [`DummyKernelMatrix`](@ref) in [`buildkernelmatrix`](@ref). Use it
as `matrixdata` of an [`AssemblyStrategy`](@ref) to switch the near field off, e.g. to check
or measure the far field on its own.
"""
struct DummyEntries end

"""
    DummyKernelMatrix{T}(nrows, ncolumns)

Kernel matrix whose entries are all zero. It has no near blocks, so assembling its near
interactions neither walks the tree nor evaluates a block, and returns an empty matrix in
the selected storage.
"""
struct DummyKernelMatrix{T} <: AbstractKernelMatrix{T}
    nrows::Int
    ncolumns::Int

    function DummyKernelMatrix{T}(nrows::Integer, ncolumns::Integer) where {T}
        return new{T}(nrows, ncolumns)
    end
end

Base.size(matrix::DummyKernelMatrix) = (matrix.nrows, matrix.ncolumns)

# Adding zero entries leaves `block` unchanged.
(::DummyKernelMatrix)(block, rows, columns) = nothing
