#include <curand_kernel.h>

#include "gpu_traverse.cuh"

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_states) {
        return;
    }
    curand_init(1234, i, 0, states + i);
}

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

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::CLOSEST_PRIMITIVE, false);
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
    int *alive,
    int n_rays,
    bool nbvh_only
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    Ray ray = {ray_origins[i], ray_vectors[i]};
    StackInfo stack_info = {stack_sizes[i], stack + i * stack_limit};

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX, nbvh_only);
    masks[i] = hit.hit;
    t1[i] = hit.t1;
    t2[i] = hit.t2;
    if (hit.hit) {
        node_idxs[i] = hit.node_idx;
    } else {
        node_idxs[i] = 0;
    }

    // I am sorry
    atomicOr(alive, hit.hit);
}

CUDA_GLOBAL void bbox_raygen_entry(
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_limit,
    curandState *rand_states,
    uint32_t *leaf_idxs,
    int n_leaves,
    glm::vec3 *ray_origins,
    glm::vec3 *ray_ends,
    uint32_t *bbox_idxs,
    bool *masks,
    float *t, // value in [0, 1]
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    // ==== Generate random leaf index ==== //

    curandState *state = &rand_states[i];

    uint32_t leaf_idx = leaf_idxs[curand(state) % n_leaves];
    const BVHNode &leaf = dp.nodes[leaf_idx];


    // ==== Generate ray start and end ==== //

    glm::vec3 min = leaf.bbox.min;
    glm::vec3 max = leaf.bbox.max;
    glm::vec3 extent = max - min;
    glm::vec3 center = (max + min) * 0.5f;

    glm::vec3 p1 = glm::vec3(
        curand_uniform(state),
        curand_uniform(state), 
        curand_uniform(state)
    ) * extent + min;

    glm::vec3 p2p1 = glm::normalize(glm::vec3(
        curand_uniform(state) - 0.5f,
        curand_uniform(state) - 0.5f, 
        curand_uniform(state) - 0.5f
    ));

    HitResult bbox_hit = ray_box_intersection(Ray{p1, p2p1}, leaf.bbox);
    if (!bbox_hit.hit) {
        ray_origins[i] = p1 * 0.f;
        ray_ends[i] = p1 * 0.f;
        bbox_idxs[i] = 0;
        masks[i] = false;
        t[i] = 0;
        return;
    }
    
    glm::vec3 ray_origin = p1 + bbox_hit.t1 * p2p1;
    glm::vec3 ray_end = p1 + bbox_hit.t2 * p2p1;
    glm::vec3 ray_vector = p2p1;


    // ==== Intersect the primitives ==== //

    StackInfo st = {stack_sizes[i], stack + i * stack_limit};

    (stack + i * stack_limit)[0] = leaf_idx;

    HitResult hit = bvh_traverse(Ray{ray_origin, ray_vector}, dp, st, TraverseMode::CLOSEST_PRIMITIVE, false);

    ray_origins[i] = ray_origin;
    ray_ends[i] = ray_end;
    bbox_idxs[i] = leaf_idx;
    masks[i] = hit.hit;
    t[i] = hit.t;
}
