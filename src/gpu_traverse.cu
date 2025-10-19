#include <curand_kernel.h>

#include "gpu_traverse.cuh"

#define EPS 1e-6

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_states) {
        return;
    }
    curand_init(1234, i, 0, states + i);
}

CUDA_GLOBAL void traverse_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_out,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];

    float closest_t = 0;

    HitResult hit = bvh_traverse(ray, i_dp, true, closest_t);
    o_out.fill(i, hit);
}
