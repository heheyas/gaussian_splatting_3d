#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ inline scalar_t gaussian_kernel(scalar_t *mean, scalar_t *qvec,
                                           scalar_t *tvec, scalar_t *query) {}