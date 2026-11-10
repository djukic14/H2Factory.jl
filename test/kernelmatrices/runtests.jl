module TestKernelMatricesGroup

using Test

@testset verbose = true "Kernel matrices" begin
    include("test_kernelmatrices.jl")
end

end # module TestKernelMatricesGroup
