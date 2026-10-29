"""
    farmul!(y, A, x)
    farmul!(y, A, x, α, β)

Apply only the far-interaction part of `A`, storing the result in `y`. The five-argument form
computes `y = α * far(A) * x + β * y`.
"""
function farmul!(y::AbstractVector, A, x::AbstractVector)
    return farmul!(y, A, x, NoDiagnostics())
end

function farmul!(y::AbstractVector, A, x::AbstractVector, α::Number, β::Number)
    return farmul!(y, A, x, α, β, NoDiagnostics())
end

# Public calls own the workspace for the complete far-field traversal.
function farmul!(
    y::AbstractVector, A, x::AbstractVector, diagnostics::D
) where {D<:AbstractDiagnostics}
    return withworkspace(A) do
        return _farmul!(y, A, x, diagnostics)
    end
end

function farmul!(
    y::AbstractVector, A, x::AbstractVector, α::Number, β::Number, diagnostics::D
) where {D<:AbstractDiagnostics}
    return withworkspace(A) do
        return _farmul!(y, A, x, α, β, diagnostics)
    end
end

function _farmul!(
    y::AbstractVector, A, x::AbstractVector, diagnostics::D
) where {D<:AbstractDiagnostics}
    checkexecutiondata(A)
    return farmul!(y, A, maptrait(A), farmulmode(A), x, diagnostics)
end

function _farmul!(
    y::AbstractVector, A, x::AbstractVector, α::Number, β::Number, diagnostics::D
) where {D<:AbstractDiagnostics}
    checkexecutiondata(A)
    return farmul!(y, A, maptrait(A), farmulmode(A), x, α, β, diagnostics)
end

function farmul!(
    y::AbstractVector, A, ::AbstractMapTrait, ::Any, x::AbstractVector, diagnostics::D
) where {D<:AbstractDiagnostics}
    # Some receiving nodes have no far contribution and therefore never write their slice.
    fill!(y, zero(eltype(y)))
    aggregate!(A, x, diagnostics)
    disaggregate!(y, A, diagnostics)
    return y
end

function farmul!(
    y::AbstractVector,
    A,
    ::AbstractMapTrait,
    ::Any,
    x::AbstractVector,
    α::Number,
    β::Number,
    diagnostics::D,
) where {D<:AbstractDiagnostics}
    aggregate!(A, x, diagnostics)
    disaggregate!(y, A, α, β, diagnostics)
    return y
end

"""
    farmul(A, x)

Return the far-interaction product of `A` and `x` in a new vector.
"""
function farmul(A, x::AbstractVector)
    y = Vector{promote_type(eltype(x), eltype(A))}(undef, size(A, 1))
    return farmul!(y, A, x)
end

"""
    farinteractions(A, A_H2)
    farinteractions(A, tree, isfar)

Extract the dense far-interaction entries of matrix-like object `A`. The map form respects the
orientation of transpose and adjoint views.
"""
function farinteractions end

function farinteractions(A, A_H2)
    tree = H2Trees.tree(A_H2)
    return _orientedfarinteractions(
        A, tree, H2Trees.treetrait(tree), H2Trees.isfar(A_H2), A_H2
    )
end

# A single tree uses the same side in every orientation.
function _orientedfarinteractions(A, tree, trait::H2Trees.AbstractTreeTrait, isfar, A_H2)
    return farinteractions(A, tree, trait, isfar)
end

function _orientedfarinteractions(A, tree, ::H2Trees.isBlockTree, isfar, A_H2)
    testtree, trialtree = blocktreesides(A_H2, tree)
    nodes = collect(Int, H2Trees.leaves(testtree))
    rowgroups = [H2Trees.values(testtree, node) for node in nodes]
    colgroups = [
        H2Trees.farnodevalues(trialtree, testtree, node; isfar=isfar) for node in nodes
    ]
    return _denseinteractionmatrix(A, rowgroups, colgroups)
end

function farinteractions(A, tree, isfar)
    return farinteractions(A, tree, H2Trees.treetrait(tree), isfar)
end

function farinteractions(A, tree, ::H2Trees.AbstractTreeTrait, isfar)
    nodes = collect(Int, H2Trees.leaves(tree))
    rowgroups = [H2Trees.values(tree, node) for node in nodes]
    colgroups = [H2Trees.farnodevalues(tree, node; isfar=isfar) for node in nodes]
    return _denseinteractionmatrix(A, rowgroups, colgroups)
end

function farinteractions(A, tree, ::H2Trees.isBlockTree, isfar)
    testtree = H2Trees.testtree(tree)
    trialtree = H2Trees.trialtree(tree)
    nodes = collect(Int, H2Trees.leaves(testtree))
    rowgroups = [H2Trees.values(testtree, node) for node in nodes]
    colgroups = [
        H2Trees.farnodevalues(trialtree, testtree, node; isfar=isfar) for node in nodes
    ]
    return _denseinteractionmatrix(A, rowgroups, colgroups)
end
