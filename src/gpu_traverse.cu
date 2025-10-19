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

CUDA_GLOBAL void ray_query_entry(
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

    HitResult hit = ray_query(ray, i_dp, true);
    o_out.fill(i, hit);
}

CUDA_GLOBAL void point_query_entry(
    const glm::vec3 *i_points,
    const BVHDataPointers i_dp,
    SDFHitResults o_out,
    int n_points
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    glm::vec3 point = i_points[i];

    SDFHitResult hit = point_query(point, i_dp);
    o_out.fill(i, hit);
}
