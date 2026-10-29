"""
    nearmul!(y, A, x)
    nearmul!(y, A, x, α, β)

Multiply `x` by the near-interaction matrix of `A`, storing the result in `y`. The five-argument
form computes `y = α * nearinteractions(A) * x + β * y`.
"""
function nearmul!(y::AbstractVector, A, x::AbstractVector)
    return LinearAlgebra.mul!(y, nearinteractions(A), x)
end

function nearmul!(y::AbstractVector, A, x::AbstractVector, α::Number, β::Number)
    return LinearAlgebra.mul!(y, nearinteractions(A), x, α, β)
end

"""
    nearmul(A, x)

Return `nearinteractions(A) * x` in a newly allocated vector.
"""
function nearmul(A, x::AbstractVector)
    y = Vector{promote_type(eltype(A), eltype(x))}(undef, size(A, 1))
    return nearmul!(y, A, x)
end

"""
    nearinteractions(A, A_H2)

Extract the near-interaction entries of matrix-like object `A` using the oriented trees and near
predicate of `A_H2`. Transpose and adjoint views swap the row and column sides when needed.
"""
function nearinteractions(A, A_H2)
    tree = H2Trees.tree(A_H2)
    return _orientednearinteractions(
        A, tree, H2Trees.treetrait(tree), H2Trees.isnear(A_H2), A_H2
    )
end

# A single tree uses the same side in every orientation.
function _orientednearinteractions(A, tree, trait::H2Trees.AbstractTreeTrait, isnear, A_H2)
    return nearinteractions(A, tree, trait, isnear)
end

function _orientednearinteractions(A, tree, ::H2Trees.isBlockTree, isnear, A_H2)
    rowside, colside = blocktreesides(A_H2, tree)
    rowgroups, colgroups = H2Trees.nearinteractions(rowside, colside; isnear=isnear)
    return _sparseinteractionmatrix(A, rowgroups, colgroups)
end

function nearinteractions(A, tree, isnear)
    return nearinteractions(A, tree, H2Trees.treetrait(tree), isnear)
end

function _interactiontriplets(A, rowgroups, colgroups)
    @assert length(rowgroups) == length(colgroups)

    rows = [collect(group) for group in rowgroups]
    cols = [unique!(collect(group)) for group in colgroups]
    counts = [length(rows[i]) * length(cols[i]) for i in eachindex(rows)]
    total = sum(counts)

    I = Vector{Int}(undef, total)
    J = Vector{Int}(undef, total)
    V = Vector{eltype(A)}(undef, total)
    offsets = cumsum([0; counts])

    @threads for group in eachindex(rows)
        index = offsets[group] + 1
        for col in cols[group]
            for row in rows[group]
                I[index] = row
                J[index] = col
                V[index] = A[row, col]
                index += 1
            end
        end
    end

    return I, J, V
end

function _sparseinteractionmatrix(A, rowgroups, colgroups)
    I, J, V = _interactiontriplets(A, rowgroups, colgroups)
    return sparse(I, J, V, size(A)...)
end

function _denseinteractionmatrix(A, rowgroups, colgroups)
    matrix = zeros(eltype(A), size(A))

    @threads for group in eachindex(rowgroups)
        rows = rowgroups[group]
        cols = unique!(collect(colgroups[group]))
        isempty(rows) && continue
        isempty(cols) && continue
        matrix[rows, cols] = A[rows, cols]
    end

    return matrix
end

function nearinteractions(A, tree, ::H2Trees.AbstractTreeTrait, isnear)
    rowgroups, colgroups = H2Trees.nearinteractions(tree; isnear=isnear)
    return _sparseinteractionmatrix(A, rowgroups, colgroups)
end

function nearinteractions(A, tree, ::H2Trees.isBlockTree, isnear)
    rowgroups, colgroups = H2Trees.nearinteractions(
        H2Trees.testtree(tree), H2Trees.trialtree(tree); isnear=isnear
    )
    return _sparseinteractionmatrix(A, rowgroups, colgroups)
end
