module TestMaps

using Test

@testset verbose = true "maps" begin
    include("test_traits.jl")
    include("test_workspace.jl")
end

end
