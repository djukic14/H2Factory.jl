module TestH2BlockSparseGroup

using Test

@testset verbose = true "H2BlockSparse" begin
    include("test_smoke.jl")
end

end # module TestH2BlockSparseGroup
