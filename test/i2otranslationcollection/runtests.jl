module TestI2OCollectionGroup

using Test
# TestUtils-dependent includes stay commented until H2FactoryTestUtils is registered.

@testset verbose = true "I2OCollection" begin
    include("test_dummycollection.jl")
    # include("test_i2ocollection.jl")
    include("test_symmetrycollection.jl")
    include("test_toeplitzpermutation.jl")
    include("test_componentaction.jl")
    include("test_reducedtoeplitz.jl")
    include("test_reduceddense.jl")
    include("test_compositioncontract.jl")
end

end # module TestI2OCollectionGroup
