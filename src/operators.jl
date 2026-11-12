# Unitary, centered 2-D Fourier transforms: the link between image and k-space.
# Dividing by √N in one direction and multiplying by √N in the other keeps the norm,
# so neither transform changes the size of what it is given.

"""
    fft2c(x)

Image → k-space. The 2-D DFT of `x`, divided by √N with N = `length(x)`, so that the
norm is unchanged. The centre of k-space sits at index `N_d ÷ 2 + 1` along each
dimension of size `N_d` (49 for 96), as in scanner data.
"""
fft2c(x::AbstractMatrix) = fftshift(fft(ifftshift(x))) ./ sqrt(length(x))

"""
    ifft2c(y)

k-space → image. The inverse of [`fft2c`](@ref): the 2-D inverse DFT of `y`, multiplied
by √N with N = `length(y)`, so that the norm is unchanged.
"""
ifft2c(y::AbstractMatrix) = fftshift(ifft(ifftshift(y))) .* sqrt(length(y))
