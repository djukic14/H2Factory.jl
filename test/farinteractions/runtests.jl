module TestFarInteractionsGroup

using Test
# TestUtils-dependent includes stay commented until H2FactoryTestUtils is registered.

@testset verbose = true "Far interactions" begin
    # include("test_farmul.jl")
end

@testset verbose = true "Translation application protocol" begin
    # include("test_translationprotocol.jl")
end

@testset verbose = true "I2O protocol (coupling + modulation composition)" begin
    include("test_i2oprotocol.jl")
end

@testset verbose = true "Level-transfer protocol" begin
    include("test_leveltranslator.jl")
end

@testset verbose = true "Level-transfer traversal (LevelTranslator through farmul!)" begin
    # include("test_leveltransfer_traversal.jl")
end

@testset verbose = true "Traversal chunk schedules" begin
    # include("test_schedules.jl")
end

@testset verbose = true "Aggregation/disaggregation harness" begin
    # include("test_component_harness.jl")
end

@testset verbose = true "Adjoint leaf/α conjugations" begin
    # include("test_adjointconjugations.jl")
end

@testset verbose = true "Transfer side ownership" begin
    # include("test_sideownership.jl")
end

@testset verbose = true "Leaf-multiply scratch channel" begin
    # include("test_leafchannel.jl")
end

@testset verbose = true "Rectangular DOF coverage" begin
    # include("test_rectangular.jl")
end

@testset verbose = true "Interaction extraction harness" begin
    # include("test_interaction_extraction_harness.jl")
    # include("test_factorization.jl")
end

@testset verbose = true "Aggregation/disaggregation harness (Toeplitz)" begin
    # include("test_component_harness_toeplitz.jl")
end

@testset verbose = true "Toeplitz far-field I2O" begin
    # include("test_toeplitz.jl")
end

@testset verbose = true "Toeplitz far-field I2O (three-dimensional)" begin
    # include("test_toeplitz3d.jl")
end

@testset verbose = true "Aggregation/disaggregation harness (symmetry-reduced I2O)" begin
    # include("test_component_harness_symmetry.jl")
end

end # module TestFarInteractionsGroup
