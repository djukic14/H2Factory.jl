module TestNearInteractionsGroup

using Test
# TestUtils-dependent includes stay commented until H2FactoryTestUtils is registered.

@testset verbose = true "Near interactions" begin
    # include("test_nearinteractions.jl")
    include("test_strategies.jl")
    include("test_strategycontract.jl")
end

end # module TestNearInteractionsGroup
