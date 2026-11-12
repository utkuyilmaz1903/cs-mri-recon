"""
    CSMRIRecon

Compressed-sensing MRI reconstruction written from scratch: a simulated single-coil
Cartesian forward model, wavelet sparsity and ISTA/FISTA solvers.
"""
module CSMRIRecon

using FFTW: fft, ifft, fftshift, ifftshift

export fft2c, ifft2c

include("operators.jl")

end # module CSMRIRecon
