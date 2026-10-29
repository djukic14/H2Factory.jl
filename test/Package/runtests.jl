module TestPackageGroup

using Test

@testset verbose = true "Package" begin
    include("publicapi.jl")
end

end # module TestPackageGroup
