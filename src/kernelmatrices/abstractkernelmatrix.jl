"""
    AbstractKernelMatrix{T}

Base type for lazily evaluated matrices with scalar type `T`. Entries are computed on
request instead of being stored.

Concrete kernel matrices implement:

  - `Base.size(matrix)`: the full `(rows, columns)` size.
  - `matrix(block, rows, columns)`: add the entries `matrix[rows, columns]` to `block`.

`rows` and `columns` are global indices. Each may be an integer or a collection of integers;
`block` keeps a two-dimensional shape of `(length(rows), length(columns))` in both cases.
The call accumulates, so callers that need the plain entries pass zero-initialized storage.
"""
abstract type AbstractKernelMatrix{T} end

Base.eltype(::Type{<:AbstractKernelMatrix{T}}) where {T} = T

function Base.size(matrix::AbstractKernelMatrix, dim::Integer)
    return size(matrix)[dim]
end

# Without this fallback a missing block method surfaces as a `MethodError` on a callable
# object, which does not name the protocol the subtype failed to implement.
function (matrix::AbstractKernelMatrix)(block, rows, columns)
    return error(
        "$(typeof(matrix)) does not implement block evaluation. Define " *
        "`(matrix::$(nameof(typeof(matrix))))(block, rows, columns)` to add the entries " *
        "`matrix[rows, columns]` to `block`.",
    )
end

"""
    buildkernelmatrix(operator, testspace, trialspace, matrixdata) -> AbstractKernelMatrix

Build the kernel matrix of `operator` between `testspace` and `trialspace`.

`matrixdata` selects how entries are evaluated, e.g. a quadrature strategy. Packages
providing a kernel family implement a method for their operator, space and data types.
There is no permissive default: an unsupported combination raises an error naming the
missing method.
"""
function buildkernelmatrix(operator, testspace, trialspace, matrixdata)
    return error(
        "buildkernelmatrix is not implemented for operator::$(typeof(operator)), " *
        "testspace::$(typeof(testspace)), trialspace::$(typeof(trialspace)), " *
        "matrixdata::$(typeof(matrixdata)). Load the extension providing this kernel " *
        "family or define `H2Factory.buildkernelmatrix` for these types.",
    )
end

"""
    defaultmatrixdata(operator, testspace, trialspace)

The matrix data to pass to [`buildkernelmatrix`](@ref) when the caller selects none. The
default is `nothing`, which point kernels take. With BEAST loaded, a BEAST operator defaults
to `BEAST.defaultquadstrat(operator, testspace, trialspace)`.
"""
defaultmatrixdata(operator, testspace, trialspace) = nothing

"""
    localkernelmatrix(matrix, rows, columns) -> AbstractKernelMatrix

Prepare `matrix` for repeated row and column sampling within one block.

Rows are sampled against exactly `columns` and columns against exactly `rows`. Either may
be `nothing` when that side is never sampled. The result accepts the original global
indices and has the `size` and `eltype` of `matrix`.

The default returns `matrix` unchanged. A backend may return a wrapper that precomputes
block-dependent data. Such a wrapper may hold mutable scratch and must not be shared between
concurrent tasks.

With BEAST loaded, a [`BEASTKernelMatrix`](@ref) with `BEAST.DoubleNumQStrat` returns a
wrapper that integrates a sampled row only over the elements of `columns`, and a sampled
column only over the elements of `rows`. Other quadrature strategies keep the default.
"""
localkernelmatrix(matrix, rows, columns) = matrix
