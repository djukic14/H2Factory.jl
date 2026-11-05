module TestPackageGroup

using Test

@testset verbose = true "Package" begin
    include("publicapi.jl")
    include("test_timing.jl")
end

end # module TestPackageGroup
