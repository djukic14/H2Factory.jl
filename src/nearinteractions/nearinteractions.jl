"""
    assemblenearinteractions(operator, testspace, trialspace, tree, assemblystrategy, interactionstrategy; verbose=false)
    assemblenearinteractions(kernelmatrix, tree, assemblystrategy, interactionstrategy; verbose=false)

Assemble the near interactions of `tree` into the format selected by
`storage(assemblystrategy)`.

The first form builds the kernel matrix with [`buildkernelmatrix`](@ref) from
`matrixdata(assemblystrategy)`. The near pairs follow `nearcriterion(interactionstrategy)`,
the predicate the far-field plans use. The stored entries have type `eltype(kernelmatrix)`.
"""
function assemblenearinteractions(
    operator,
    testspace,
    trialspace,
    tree,
    assemblystrategy::AbstractAssemblyStrategy,
    interactionstrategy::AbstractInteractionStrategy;
    verbose=false,
)
    kernelmatrix = buildkernelmatrix(
        operator, testspace, trialspace, matrixdata(assemblystrategy)
    )
    return assemblenearinteractions(
        kernelmatrix, tree, assemblystrategy, interactionstrategy; verbose=verbose
    )
end

function assemblenearinteractions(
    kernelmatrix::AbstractKernelMatrix,
    tree,
    assemblystrategy::AbstractAssemblyStrategy,
    interactionstrategy::AbstractInteractionStrategy;
    verbose=false,
)
    rows, columns = nearblocks(kernelmatrix, tree, interactionstrategy)
    return assembleblocks(
        kernelmatrix,
        rows,
        columns,
        storage(assemblystrategy),
        scheduler(assemblystrategy);
        verbose=verbose,
    )
end

"""
    nearblocks(kernelmatrix, tree, interactionstrategy) -> (rows, columns)

Return the global row and column indices of the near blocks of `kernelmatrix`.

The default walks `tree` with `nearcriterion(interactionstrategy)`. A kernel matrix that
knows its blocks without the tree adds a method; a [`DummyKernelMatrix`](@ref) has none and
never touches the tree.
"""
function nearblocks(::AbstractKernelMatrix, tree, interactionstrategy)
    isnear = nearcriterion(interactionstrategy)(tree)
    return H2Trees.nearinteractions(tree; isnear=isnear, extractselfvalues=false)
end

nearblocks(::DummyKernelMatrix, tree, interactionstrategy) = (Vector{Int}[], Vector{Int}[])
