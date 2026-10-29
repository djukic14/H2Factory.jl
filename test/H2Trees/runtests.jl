module TestH2TreesGroup

using Test

@testset verbose = true "H2Trees" begin
    include("plans/test_allocatemoments.jl")
end

end # module TestH2TreesGroup
