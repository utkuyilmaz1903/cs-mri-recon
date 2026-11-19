# Tests for src/operators.jl: fft2c, ifft2c and undersampling_mask.
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

    @testset "undersampling_mask" begin
        rng = StableRNG(1)
        # A square size, and one with unequal sides to catch rows and columns mixed up.
        dims_96x96 = (96, 96)
        dims_7x5 = (7, 5)
        # R = 4: the scan gets four times faster.
        mask_96x96 = undersampling_mask(rng, dims_96x96, 4)
        mask_7x5 = undersampling_mask(rng, dims_7x5, 4)

        @test size(mask_96x96) == dims_96x96
        @test size(mask_7x5) == dims_7x5

        # The center block is always measured: profiles 46 to 51, or just 3 in the 7x5.
        @test all(mask_96x96[:, 46:51])
        @test all(mask_7x5[:, 3])

        # Whole profiles only: if all rows are equal, every column is all true or all false.
        @test allequal(eachrow(mask_96x96))
        @test allequal(eachrow(mask_7x5))

        # Exactly n / R profiles: 96 / 4 = 24, and 5 / 4 rounds to 1.
        @test count(mask_96x96[1, :]) == 24
        @test count(mask_7x5[1, :]) == 1

        # The same seed gives the same mask.
        @test undersampling_mask(StableRNG(2), dims_96x96, 4) ==
            undersampling_mask(StableRNG(2), dims_96x96, 4)

        # R = 1 measures everything; R < 1 would need more profiles than the table has.
        @test all(undersampling_mask(StableRNG(1), dims_96x96, 1))
        @test_throws ArgumentError undersampling_mask(StableRNG(1), dims_96x96, 0.5)

        # Profiles near the center are drawn more often: over 1000 seeds, profile 52
        # (3 from the center) should be measured more often than 73 (halfway out).
        times_52 = 0
        times_73 = 0
        for seed in 1:1000
            mask = undersampling_mask(StableRNG(seed), dims_96x96, 4)
            times_52 += mask[1, 52]
            times_73 += mask[1, 73]
        end
        @test times_52 > times_73
    end

end
