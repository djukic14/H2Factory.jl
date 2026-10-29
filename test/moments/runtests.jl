module TestMomentsGroup

using Test
# TestUtils-dependent includes stay commented until H2FactoryTestUtils is registered.

@testset verbose = true "Moment utils" begin
    include("test_momentsutil.jl")
    include("test_momentcollectionutil.jl")
    include("test_creation.jl")
    include("test_linking.jl")
    include("test_storer.jl")
    # include("test_aliasing.jl")
    include("test_toeplitzmul.jl")
    include("test_generateddocs.jl")
end

end # module TestMomentsGroup
