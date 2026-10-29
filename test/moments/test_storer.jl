module TestStorer

using Test
using StaticArrays
using BEAST, CompScienceMeshes
using H2Factory
using H2Trees

@testset "StoreAsIs I" begin
    storer = StoreAsIs(SVector{1}([zeros(3, 3)]), nothing)

    r = randn(3, 3)
    for i in eachindex(axes(r, 1)), j in eachindex(axes(r, 2))
        storer(r[i, j], nothing, i, j)
    end
    @test storer.Fs[1] ≈ r

    storer = StoreAsIs([zeros(3, 3), zeros(3, 3)], nothing)

    r1 = randn(3, 3)
    r2 = randn(3, 3)
    for i in eachindex(axes(r1, 1)), j in eachindex(axes(r1, 2))
        storer([r1[i, j], r2[i, j]], nothing, i, j)
    end

    @test storer.Fs[1] ≈ r1
    @test storer.Fs[2] ≈ r2
end

@testset "StoreAsIs II" begin
    struct FakeOperator end
    function H2Factory.storetrait(::Type{FakeOperator})
        return H2Factory.StoreAsIs
    end
    op = FakeOperator()
    storer = buildstorer(SVector{1}([zeros(3, 3)]), op, nothing)

    r = randn(3, 3)
    for i in eachindex(axes(r, 1)), j in eachindex(axes(r, 2))
        storer(r[i, j], nothing, i, j)
    end
    @test storer.Fs[1] ≈ r

    storer = buildstorer([zeros(3, 3), zeros(3, 3)], op, nothing)

    r1 = randn(3, 3)
    r2 = randn(3, 3)
    for i in eachindex(axes(r1, 1)), j in eachindex(axes(r1, 2))
        storer([r1[i, j], r2[i, j]], nothing, i, j)
    end

    @test storer.Fs[1] ≈ r1
    @test storer.Fs[2] ≈ r2
end

@testset "StoreAsIs III" begin
    λ = 1.0
    m = meshsphere(λ, λ / 10)
    X = raviartthomas(m)

    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0))

    Fs = Vector{Vector{Matrix{Float64}}}(undef, length(H2Trees.leaves(tree)))
    Fscomparison = randn(2, numfunctions(X))

    for (i, leaf) in enumerate(H2Trees.leaves(tree))
        Fs[i] = [
            zeros(1, length(H2Trees.values(tree, leaf))),
            zeros(1, length(H2Trees.values(tree, leaf))),
        ]
    end

    storer = H2Factory.StoreAsIs(Fs, nothing, tree)

    leafindex = Dict(zip(H2Trees.leaves(tree), 1:length(H2Trees.leaves(tree))))
    for leaf in H2Trees.leaves(tree)
        for value in H2Trees.values(tree, leaf)
            storer(Fscomparison[:, value], leafindex[leaf], 1, value)
        end
    end

    for (i, leaf) in enumerate(H2Trees.leaves(tree))
        for (j, value) in enumerate(H2Trees.values(tree, leaf))
            @test Fs[i][1][1, j] ≈ Fscomparison[1, value]
            @test Fs[i][2][1, j] ≈ Fscomparison[2, value]
        end
    end

    Fs = SVector{1,Vector{Float64}}([zeros(numfunctions(X))])
end

@testset "StoreAsIs IV" begin
    struct FakeOperator end
    function H2Factory.storetrait(::Type{FakeOperator})
        return H2Factory.StoreAsIs
    end
    op = FakeOperator()

    λ = 1.0
    m = meshsphere(λ, λ / 10)
    X = raviartthomas(m)

    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0))

    Fs = Vector{Vector{Matrix{Float64}}}(undef, length(H2Trees.leaves(tree)))
    Fscomparison = randn(2, numfunctions(X))

    for (i, leaf) in enumerate(H2Trees.leaves(tree))
        Fs[i] = [
            zeros(1, length(H2Trees.values(tree, leaf))),
            zeros(1, length(H2Trees.values(tree, leaf))),
        ]
    end

    storer = H2Factory.buildstorer(Fs, op, nothing, tree)

    leafindex = Dict(zip(H2Trees.leaves(tree), 1:length(H2Trees.leaves(tree))))
    for leaf in H2Trees.leaves(tree)
        for value in H2Trees.values(tree, leaf)
            storer(Fscomparison[:, value], leafindex[leaf], 1, value)
        end
    end

    for (i, leaf) in enumerate(H2Trees.leaves(tree))
        for (j, value) in enumerate(H2Trees.values(tree, leaf))
            @test Fs[i][1][1, j] ≈ Fscomparison[1, value]
            @test Fs[i][2][1, j] ≈ Fscomparison[2, value]
        end
    end

    Fs = SVector{1,Vector{Float64}}([zeros(numfunctions(X))])
end

@testset "StoreAsIs flat storer accepts a chunk id" begin
    # A storer built without an `indexinleafbox` holds one flat set of moments, but
    # `assemblemoments!` still passes a real `cluster::Int` -- the chunk id of the artificial
    # per-thread partition. That must land exactly where `cluster === nothing` would, and the
    # chunk id itself must be ignored rather than used as an index.
    r1 = randn(3, 3)
    r2 = randn(3, 3)

    flat = StoreAsIs([zeros(3, 3), zeros(3, 3)], nothing)
    chunked = StoreAsIs([zeros(3, 3), zeros(3, 3)], nothing)

    for i in eachindex(axes(r1, 1)), j in eachindex(axes(r1, 2))
        flat([r1[i, j], r2[i, j]], nothing, i, j)
        # a different chunk id per entry -- none of them may change where the value lands
        chunked([r1[i, j], r2[i, j]], 1 + (i + j) % 4, i, j)
    end

    @test chunked.Fs[1] ≈ r1
    @test chunked.Fs[2] ≈ r2
    @test chunked.Fs[1] ≈ flat.Fs[1]
    @test chunked.Fs[2] ≈ flat.Fs[2]

    # and the single-component specialisation, which takes a scalar rather than a vector
    r = randn(3, 3)
    single = StoreAsIs(SVector{1}([zeros(3, 3)]), nothing)
    for i in eachindex(axes(r, 1)), j in eachindex(axes(r, 2))
        single(r[i, j], 1 + (i + j) % 4, i, j)
    end
    @test single.Fs[1] ≈ r
end

end # module TestStorer
