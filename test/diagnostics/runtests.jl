module TestDiagnosticsGroup

using Test

@testset verbose = true "Diagnostics" begin
    include("test_diagnostics.jl")
end

end # module TestDiagnosticsGroup
