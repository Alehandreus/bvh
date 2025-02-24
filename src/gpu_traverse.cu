#include <curand_kernel.h>

#include "gpu_traverse.cuh"

CUDA_GLOBAL void closest_primitive_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors,
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_limit,
    bool *masks,
    float *t,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    Ray ray = {ray_origins[i], ray_vectors[i]};
    StackInfo stack_info = {stack_sizes[i], stack + i * stack_limit};

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::CLOSEST_PRIMITIVE);
    masks[i] = hit.hit;
    t[i] = hit.t;
}

CUDA_GLOBAL void another_bbox_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors,
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_limit,
    bool *masks,
    uint32_t *node_idxs,
    float *t1,
    float *t2,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    Ray ray = {ray_origins[i], ray_vectors[i]};
    StackInfo stack_info = {stack_sizes[i], stack + i * stack_limit};

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX);
    masks[i] = hit.hit;
    node_idxs[i] = hit.node_idx;
    t1[i] = hit.t1;
    t2[i] = hit.t2;
}

CUDA_GLOBAL void segments_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors, // ray_origins and ray_origins + ray_vectors are segment edges
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_limit,
    bool *segments,
    int n_rays,
    int n_segments
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    Ray ray = {ray_origins[i], ray_vectors[i]};
    StackInfo stack_info = {stack_sizes[i], stack + i * stack_limit};
    bool *cur_segments = segments + i * n_segments;

    float eps = 1e-6;

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX);
    while (hit.hit) {                
        float t1 = hit.t1;
        float t2 = hit.t2;

        t1 = fmaxf(t1, -eps);
        t2 = fmaxf(t2, -eps);

        uint32_t segment1 = (uint32_t) ((t1 - eps) * n_segments);
        uint32_t segment2 = (uint32_t) ((t2 + eps) * n_segments) + 1;

        segment1 = fmaxf(segment1, 0u);
        segment1 = fminf(segment1, (uint32_t) n_segments);

        segment2 = fmaxf(segment2, 1u);
        segment2 = fminf(segment2, (uint32_t) n_segments);

        for (int j = segment1; j < segment2; j++) {
            cur_segments[j] = true;
        }

        hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX);
    };    
}
