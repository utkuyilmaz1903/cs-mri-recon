# Pieces of the forward model: the unitary, centered 2-D FFT pair that links image and
# k-space, and the mask that picks which k-space profiles are measured.

"""
    fft2c(x)

Image to k-space. The 2-D DFT of `x`, divided by sqrt(N) with N = `length(x)`, so that
the norm is unchanged. The center of k-space sits at index `div(N_d, 2) + 1` along each
dimension of size `N_d` (49 for 96), as in scanner data.
"""
fft2c(x::AbstractMatrix) = fftshift(fft(ifftshift(x))) ./ sqrt(length(x))

"""
    ifft2c(y)

k-space to image. The inverse of [`fft2c`](@ref): the 2-D inverse DFT of `y`, multiplied
by sqrt(N) with N = `length(y)`, so that the norm is unchanged.
"""
ifft2c(y::AbstractMatrix) = fftshift(ifft(ifftshift(y))) .* sqrt(length(y))

"""
    undersampling_mask(rng, dims, R; center_fraction = 1 / 4, decay = 2)

Pick which profiles (columns) of a k-space table of size `dims` to measure, about one in
`R`, using the random number generator `rng`. Returns a `Bool` table of size `dims` whose
columns are all `true` (measured) or all `false` (skipped), so multiplying a table by it
keeps the measured profiles and zeroes the rest.

A block of `center_fraction` of the measured profiles around the k-space center is always
measured. Its size is rounded up, so for any `center_fraction > 0` it holds at least the
center profile. The others are drawn at random without repeats; a profile at distance `d`
from the center has weight `(1 - d / D)^decay`, where `D` is one more than the largest
distance. Both defaults are our choice: a quarter for the center leaves most profiles to
the draw, and with `decay = 2` a profile halfway out has about a quarter of the center's
weight.
"""
function undersampling_mask(rng, dims, R; center_fraction = 1 / 4, decay = 2)
    R >= 1 || throw(ArgumentError("R must be at least 1, got $R"))
    n_profiles = dims[2]
    n_measured = round(Int, n_profiles / R)
    # Rounded up, so the center profile is measured even when very few profiles are.
    n_center = ceil(Int, center_fraction * n_measured)
    n_random = n_measured - n_center

    # The center profile sits where fft2c puts it: 49 for 96.
    center = div(n_profiles, 2) + 1
    start = center - div(n_center, 2)
    measured = collect(start:(start + n_center - 1))

    # D is one past the farthest profile, so even that one keeps a small weight;
    # with R = 1 every profile has to be drawn.
    distance = abs.((1:n_profiles) .- center)
    D = maximum(distance) + 1
    weight = (1 .- distance ./ D) .^ decay
    weight[measured] .= 0

    # Laid end to end, the weights cover 0 to their sum. A random point in there lands
    # on profile j with chance weight[j] / sum; a drawn profile then drops out.
    for _ in 1:n_random
        cumulative = cumsum(weight)
        point = rand(rng) * cumulative[end]
        j = findfirst(cumulative .> point)
        push!(measured, j)
        weight[j] = 0
    end

    mask = falses(dims)
    mask[:, measured] .= true
    return mask
end
