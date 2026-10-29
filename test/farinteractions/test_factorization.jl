module TestFactorization

using Test
using LinearAlgebra
using StaticArrays
using H2Factory
using H2Trees
using H2FactoryTestUtils

include("component_harness_fixtures.jl")

# Phase 0 of the Part E "factorized Petrov operator" refactor: the ORACLE every later phase is
# gated against. It pins the operator as a composition of the two stages the traversal actually
# exposes, and pins how those stages relate under transposition.
#
#     S1 = the `aggregate!` stage       S2 = the `disaggregate!` stage
#     forward:    A_far == S2 * S1
#     transpose:  S1' == transpose(S2)   and   S2' == transpose(S1)
#
# The second line is the substantive one. A composite check (`A_farᵀ == transpose(A_far)`) passes
# just as happily when the transposed traversal reuses forward kernels under swapped names -- the
# exact shape of the two orientation bugs found in `farinteractions`/`nearinteractions`, both of
# which were SILENT whenever test and trial DOF counts matched. Comparing the stages pairwise
# instead asserts that the transposed traversal's FIRST stage is the transpose of the forward
# traversal's LAST stage, which reuse-with-swapped-names cannot satisfy.

# Which moments carry data BETWEEN the two calls depends on the aggregation mode and the
# orientation, because the fused stage moves:
#
#   * aggregation plan NOT translating (pure up-pass)   -> it writes its own aggregation side
#   * aggregation plan IS translating (fused translate) -> it writes its RECEIVING side
#
# In `AggregateMode` the forward aggregation is pure and the disaggregation is fused; transposed,
# they swap. Resolving through the map's own oriented accessors covers every combination.
function intermediatedict(M)
    plan = H2Factory.trialaggregationplan(M)
    return if H2Trees.istranslatingplan(plan)
        H2Factory.testmoments(M)
    else
        H2Factory.trialmoments(M)
    end
end

# Only moments that actually SURVIVE between the two calls are part of the interface. Transient
# moments alias through the shared even/odd pools and are dead once their level is past, so
# including them would not be a linear map at all.
function intermediatenodes(M)
    plan = H2Factory.trialaggregationplan(M)
    nodes = Int[]
    if H2Trees.istranslatingplan(plan)
        for level in H2Trees.aggregationlevels(plan)
            append!(nodes, H2Trees.receivingnodes(plan, level))
        end
    else
        for level in H2Trees.aggregationlevels(plan),
            node in H2Trees.aggregationnodes(plan, level)

            H2Trees.storenode(plan, node) && push!(nodes, node)
        end
    end
    return sort!(unique!(nodes))
end

function zeroallmoments!(A)
    for (_, m) in H2Factory.trialmoments(A)
        m[1] .= 0
    end
    for (_, m) in H2Factory.testmoments(A)
        m[1] .= 0
    end
    return A
end

flattenintermediate(M, nodes) = vcat((vec(intermediatedict(M)[n][1]) for n in nodes)...)

function injectintermediate!(M, nodes, v)
    offset = 0
    for n in nodes
        m = intermediatedict(M)[n][1]
        m .= v[(offset + 1):(offset + length(m))]
        offset += length(m)
    end
    return M
end

# S1: x -> intermediate. `A` is the forward map (owns the moment storage), `M` the oriented map.
function stage1matrix(A, M, nodes, T)
    cols = Vector{Vector{T}}()
    for j in 1:size(M, 2)
        zeroallmoments!(A)
        e = zeros(T, size(M, 2))
        e[j] = 1
        H2Factory.aggregate!(M, e)
        push!(cols, flattenintermediate(M, nodes))
    end
    return reduce(hcat, cols)
end

# S2: intermediate -> y.
function stage2matrix(A, M, nodes, T, nint)
    cols = Vector{Vector{T}}()
    for i in 1:nint
        zeroallmoments!(A)
        e = zeros(T, nint)
        e[i] = 1
        injectintermediate!(M, nodes, e)
        y = zeros(T, size(M, 1))
        H2Factory.disaggregate!(y, M)
        # Since Part E's adjoint-factorization slice, `farmul!`/`disaggregate!` apply NO trailing
        # vector conjugation for the adjoint (the conjugation lives on the I2O coupling operator), so
        # stage-wise extraction reproduces nothing extra.
        push!(cols, copy(y))
    end
    return reduce(hcat, cols)
end

function farmatrix(M, T)
    out = zeros(T, size(M, 1), size(M, 2))
    e = zeros(T, size(M, 2))
    y = zeros(T, size(M, 1))
    for j in 1:size(M, 2)
        fill!(e, 0)
        e[j] = 1
        H2Factory.farmul!(y, M, e)
        out[:, j] .= y
    end
    return out
end

relerr(a, b) = norm(a - b) / max(norm(b), eps())

function checkfactorization(A)
    T = eltype(A)
    nodes = intermediatenodes(A)
    nint = sum(length(intermediatedict(A)[n][1]) for n in nodes)
    @test !isempty(nodes)

    S1 = stage1matrix(A, A, nodes, T)
    S2 = stage2matrix(A, A, nodes, T, nint)
    farmatrixforward = farmatrix(A, T)

    # The operator IS the composition of its two stages.
    @test relerr(S2 * S1, farmatrixforward) < 1e-12
    @test size(S1) == (nint, size(A, 2))
    @test size(S2) == (size(A, 1), nint)

    transposemap = transpose(A)
    S1transposed = stage1matrix(A, transposemap, nodes, T)
    S2transposed = stage2matrix(A, transposemap, nodes, T, nint)

    @test relerr(S2transposed * S1transposed, transpose(farmatrixforward)) < 1e-12
    # THE POINT: each transposed stage is the transpose of the CORRESPONDING forward stage, with
    # the ends swapped. Reuse-with-swapped-names cannot satisfy this.
    @test relerr(S1transposed, transpose(S2)) < 1e-12
    @test relerr(S2transposed, transpose(S1)) < 1e-12

    # Adjoint. Part E's adjoint-factorization slice makes the adjoint conjugate each FACTOR at its
    # boundary (the I2O coupling operator via `AdjointTranslation`; level transfers are real, so their
    # adjoint is their transpose), rather than conjugating vectors/moments around the whole traversal.
    # So BOTH stage-adjoint identities now hold -- the adjoint map's stages are the adjoints of the
    # forward stages with the ends swapped. `S1' == adjoint(S2)` used to fail (~0.94) for a genuinely
    # complex coupling in `AggregateMode`; that is the completion criterion this slice closes.
    adjointmap = adjoint(A)
    S1adjoint = stage1matrix(A, adjointmap, nodes, T)
    S2adjoint = stage2matrix(A, adjointmap, nodes, T, nint)
    @test relerr(S2adjoint, adjoint(S1)) < 1e-12
    @test relerr(S1adjoint, adjoint(S2)) < 1e-12

    # Adjoint correctness itself, independent of any factorization: the composite matches, and the
    # defining inner-product identity holds.
    @test relerr(farmatrix(adjointmap, T), adjoint(farmatrixforward)) < 1e-12
    x = randn(T, size(A, 2))
    u = randn(T, size(A, 1))
    lhs = dot(u, H2Factory.farmul!(zeros(T, size(A, 1)), A, x))
    rhs = dot(H2Factory.farmul!(zeros(T, size(A, 2)), adjointmap, u), x)
    @test lhs ≈ rhs
    return true
end

@testset "Factorization oracle (Part E Phase 0): A_far == S2*S1, transposed stages swap ends" begin
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        for (couplinglabel, i2o) in (
            ("identity", H2FactoryTestUtils.identitytranslator()),
            ("complex scalar", H2FactoryTestUtils.scalartranslator(1.7 - 0.9im)),
        )
            @testset "Petrov rectangular / $(typeof(mode).name.name) / $couplinglabel" begin
                @test checkfactorization(
                    H2FactoryTestUtils.fakepetrovmap(
                        mkpts(uniformpts(64)),
                        mkpts(0.5 .+ uniformpts(48));
                        rank=2,
                        i2o=i2o,
                        testminvalues=1,
                        trialminvalues=1,
                        aggregationmode=mode,
                    ),
                )
            end
            @testset "Petrov coincident / $(typeof(mode).name.name) / $couplinglabel" begin
                @test checkfactorization(
                    H2FactoryTestUtils.fakepetrovmap(
                        mkpts(uniformpts(64)),
                        mkpts(uniformpts(64));
                        rank=2,
                        i2o=i2o,
                        testminvalues=1,
                        trialminvalues=1,
                        aggregationmode=mode,
                    ),
                )
            end
        end
    end
end

end # module TestFactorization
