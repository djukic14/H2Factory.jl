module TestTimingLogs

using Test
include("../timing.jl")
using .TestTiming: timed

@testset "Test progress logging" begin
    io = IOBuffer()
    result = timed("example"; io=io) do
        @test occursin("[test] START example", String(take!(io)))
        return 42
    end
    @test result == 42
    @test occursin("[test] DONE example elapsed=", String(take!(io)))

    @test_throws ErrorException timed(() -> error("example failure"), "failure"; io=io)
    log = String(take!(io))
    @test occursin("[test] START failure", log)
    @test occursin("[test] ERROR failure elapsed=", log)
    @test !occursin("[test] DONE failure", log)
end

end
