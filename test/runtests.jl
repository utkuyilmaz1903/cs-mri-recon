using CSMRIRecon
using Test

@testset "CSMRIRecon.jl" begin
    @test CSMRIRecon isa Module
    include("test_operators.jl")
end
