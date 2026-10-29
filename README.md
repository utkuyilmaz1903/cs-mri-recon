# cs-mri-recon

Compressed-sensing MRI reconstruction (ISTA/FISTA with wavelet sparsity), written from
scratch in Julia, on simulated single-coil Cartesian data.

**Status: work in progress.** Nothing here is finished or verified yet. This README will
describe the method, the tests and the results once they exist.

## Running the tests

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```
