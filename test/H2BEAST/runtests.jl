module TestH2BEASTGroup

using Test

@testset verbose = true "H2BEAST" begin
    include("test_wrappers.jl")
    include("test_assemblemoments_driver.jl")
    include("test_assemblemoments_potential.jl")
end

end # module TestH2BEASTGroup
