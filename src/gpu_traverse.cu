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

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
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
    TreeType tree_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    Ray ray = {ray_origins[i], ray_vectors[i]};
    StackInfo stack_info = {stack_sizes[i], stack + i * stack_limit};

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX, tree_type);
    masks[i] = hit.hit;
    node_idxs[i] = hit.node_idx;
    t1[i] = hit.t1;
    t2[i] = hit.t2;

    // I as sorry
    atomicOr(alive, hit.hit);
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

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX, TreeType::BVH);
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

        hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX, TreeType::BVH);
    };    
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

    HitResult hit = bvh_traverse(Ray{ray_origin, ray_vector}, dp, st, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);

    ray_origins[i] = ray_origin;
    ray_ends[i] = ray_end;
    bbox_idxs[i] = leaf_idx;
    masks[i] = hit.hit;
    t[i] = hit.t;
}
