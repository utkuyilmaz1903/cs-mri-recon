# Tests for fft2c and ifft2c (src/operators.jl).
# The 7x5 input is there on purpose: an odd size catches a swapped fftshift/ifftshift,
# and a non-square one catches a wrong N.

using StableRNGs, LinearAlgebra

@testset "operators.jl" begin
    rng = StableRNG(1)

    table_8x8 = randn(rng, ComplexF64, 8, 8)
    table_7x5 = randn(rng, ComplexF64, 7, 5)

    # Rounding alone leaves errors near 1e-15, while a real bug (a missing 1/sqrt(N))
    # is off by 10% or more. sqrt(eps()), about 1.5e-8 and the default rtol of
    # isapprox, sits safely in between.
    tol = sqrt(eps())

    @test isapprox(norm(fft2c(table_8x8)), norm(table_8x8); rtol = tol)
    @test isapprox(ifft2c(fft2c(table_8x8)), table_8x8; rtol = tol)
    @test isapprox(norm(ifft2c(table_8x8)), norm(table_8x8); rtol = tol)
    @test isapprox(fft2c(ifft2c(table_8x8)), table_8x8; rtol = tol)

    @test isapprox(norm(fft2c(table_7x5)), norm(table_7x5); rtol = tol)
    @test isapprox(ifft2c(fft2c(table_7x5)), table_7x5; rtol = tol)
    @test isapprox(norm(ifft2c(table_7x5)), norm(table_7x5); rtol = tol)
    @test isapprox(fft2c(ifft2c(table_7x5)), table_7x5; rtol = tol)
end
