"""
    PointKernelMatrix{T}(functor, testpoints, trialpoints)

Kernel matrix with entries `functor(testpoints[i], trialpoints[j])` of type `T`.

`testpoints` and `trialpoints` are indexable collections; their lengths give the size of the
matrix. Concurrent block evaluation is safe when `functor` is.
"""
struct PointKernelMatrix{T,Functor,TestPoints,TrialPoints} <: AbstractKernelMatrix{T}
    functor::Functor
    testpoints::TestPoints
    trialpoints::TrialPoints

    function PointKernelMatrix{T}(functor, testpoints, trialpoints) where {T}
        return new{T,typeof(functor),typeof(testpoints),typeof(trialpoints)}(
            functor, testpoints, trialpoints
        )
    end
end

function Base.size(matrix::PointKernelMatrix)
    return (length(matrix.testpoints), length(matrix.trialpoints))
end

function (matrix::PointKernelMatrix)(block, rows, columns)
    for (n, column) in enumerate(columns), (m, row) in enumerate(rows)
        block[m, n] += matrix.functor(matrix.testpoints[row], matrix.trialpoints[column])
    end
    return nothing
end

"""
    buildkernelmatrix(operator, testpoints::AbstractVector{<:SVector},
        trialpoints::AbstractVector{<:SVector}, ::Nothing) -> PointKernelMatrix

Build the [`PointKernelMatrix`](@ref) with entries
`operator(testpoints[i], trialpoints[j])`. `operator` is callable on two points and defines
`Base.eltype(operator)`, the scalar type of its values. Point kernels take no matrix data;
[`DummyEntries`](@ref) and [`RandomEntries`](@ref) give a [`DummyKernelMatrix`](@ref) and
a [`RandomKernelMatrix`](@ref) of the same size and scalar type.

A plain function has no scalar type of its own: it is taken from the first entry, with a
warning.
"""
function buildkernelmatrix(
    operator,
    testpoints::AbstractVector{<:SVector},
    trialpoints::AbstractVector{<:SVector},
    ::Nothing,
)
    T = pointscalartype(operator, testpoints, trialpoints)
    return PointKernelMatrix{T}(operator, testpoints, trialpoints)
end

function buildkernelmatrix(
    operator,
    testpoints::AbstractVector{<:SVector},
    trialpoints::AbstractVector{<:SVector},
    ::DummyEntries,
)
    T = pointscalartype(operator, testpoints, trialpoints)
    return DummyKernelMatrix{T}(length(testpoints), length(trialpoints))
end

function buildkernelmatrix(
    operator,
    testpoints::AbstractVector{<:SVector},
    trialpoints::AbstractVector{<:SVector},
    ::RandomEntries,
)
    T = pointscalartype(operator, testpoints, trialpoints)
    return RandomKernelMatrix{T}(length(testpoints), length(trialpoints))
end

# A functor states its scalar type through `eltype`.
pointscalartype(operator, testpoints, trialpoints) = eltype(operator)

# `eltype` of a plain function is `Any`, which would make every entry boxed.
function pointscalartype(operator::Function, testpoints, trialpoints)
    T = typeof(operator(first(testpoints), first(trialpoints)))
    @warn "A plain function has no scalar type; using $T, the type of its first entry. " *
        "Define a functor with `Base.eltype` to set the scalar type."
    return T
end
