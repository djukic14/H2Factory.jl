module TestH2ToeplitzGroup

using Test

@testset verbose = true "H2Toeplitz" begin
    include("test_smoke.jl")
end

end # module TestH2ToeplitzGroup
