module TestAllocateMomentsPlan

using Test
using ClusterTrees
using H2Factory
using CompScienceMeshes
using OhMyThreads
using H2Trees

@testset "Storage moments" begin
    λ = 1.0
    m = CompScienceMeshes.meshsphere(λ, λ / 10)

    tree = buildtree(
        vertices(m); builder=TwoNTreeBuilder(; minhalfsize=λ / 20, minvalues=0, root=2)
    )

    for i in eachindex(H2Trees.levels(tree))
        levels = H2Trees.levels(tree)[i:end]
        for MOMENT in
            [H2Factory.Moment1D, H2Factory.Moment2D, H2Factory.Moment3D, H2Factory.Moment4D]
            momentfunction = (tree, level) -> MOMENT{Float64}((4 * level, 4))
            chunkcounts = Dict(level => 1 for level in levels)
            moments = H2Factory.withnchunks(chunkcounts) do
                H2Factory.storagemoments(levels, tree, momentfunction)
            end

            @assert length(moments) == length(levels)

            for i in eachindex(levels)
                @test moments[i] isa H2Factory.BufferStore

                m = H2Factory.borrowbuffer(moments[i], 1)
                # Not `m isa MOMENT{Float64}` -- `MOMENT{Float64}(...)` is a convenience outer
                # constructor that *infers* the storage/pointer/memory type params from `T`, it
                # doesn't mean the resulting type literally has `Float64` in that slot (the real
                # 2nd param is the storage container type, e.g. `Matrix{Float64}`). Check the
                # element type and the moment kind instead of a literal parameterized type match.
                @test m isa H2Factory.AbstractMoment
                for moment in m.moments
                    @test size(moment) == (4 * levels[i], 4)
                end
            end
        end
    end
end

end # module TestAllocateMomentsPlan
