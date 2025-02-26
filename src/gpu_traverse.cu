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
    int *alive,
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
    const BVHNode &leaf = dp.nodes[0];


    // ==== Generate ray start and end ==== //

    glm::vec3 min = leaf.bbox.min;
    glm::vec3 max = leaf.bbox.max;
    glm::vec3 extent = max - min;

    // inflate bbox
    // min = min - 0.1f * extent;
    // max = max + 0.1f * extent;
    // extent = max - min;

    glm::vec3 p1 = glm::vec3(
        curand_uniform(state),
        curand_uniform(state), 
        curand_uniform(state)
    ) * extent + min;

    // float a1 = 0.1 + curand_uniform(state) * 0.1 + 0.8 * (curand_uniform(state) > 0.5);
    // float a2 = 0.1 + curand_uniform(state) * 0.1 + 0.8 * (curand_uniform(state) > 0.5);
    // float a3 = 0.1 + curand_uniform(state) * 0.1 + 0.8 * (curand_uniform(state) > 0.5);

    // glm::vec3 p1 = glm::vec3(a1, a2, a3) * extent + min;

    glm::vec3 p2p1 = glm::normalize(glm::vec3(
        curand_uniform(state) - 0.5f,
        curand_uniform(state) - 0.5f, 
        curand_uniform(state) - 0.5f
    ));

    // glm::vec3 p2 = glm::vec3(
    //     curand_uniform(state),
    //     curand_uniform(state), 
    //     curand_uniform(state)
    // ) * extent + min;

    // glm::vec3 p2p1 = glm::normalize(p2 - p1);

    HitResult bbox_hit = ray_box_intersection(Ray{p1, p2p1}, {min, max});
    if (!bbox_hit.hit) {
        return;
    }
    
    glm::vec3 ray_origin = p1 + bbox_hit.t1 * p2p1;
    glm::vec3 ray_end = p1 + bbox_hit.t2 * p2p1;
    glm::vec3 ray_vector = p2p1;


    // ==== Intersect the primitives ==== //

    const BVHDataPointers dp2 = {
        dp.vertices,
        dp.faces,
        dp.nodes + leaf_idx,
        dp.prim_idxs
    };

    StackInfo st = {stack_sizes[i], stack + i * stack_limit};

    // (stack + i * stack_limit)[0] = leaf_idx;

    HitResult hit = bvh_traverse(Ray{ray_origin, ray_vector}, dp, st, TraverseMode::CLOSEST_PRIMITIVE);

    float t_norm = hit.t;// / glm::length(ray_end - ray_origin); // fit t value in [0, 1]

    ray_origins[i] = ray_origin;
    ray_ends[i] = ray_end;
    masks[i] = hit.hit;
    t[i] = t_norm;

    // curandState *state = &rand_states[i];
    // StackInfo st = {stack_sizes[i], stack + i * stack_limit};

    // const BVHNode &root = dp.nodes[0];

    // glm::vec3 min = root.bbox.min;
    // glm::vec3 max = root.bbox.max;
    // glm::vec3 extent = max - min;

    // // inflate bbox
    // min = min - 0.5f * extent;
    // max = max + 0.5f * extent;
    // extent = max - min;

    // glm::vec3 p1, p2p1;
    // HitResult hit = {false, 0};

    // do {
    //     p1 = glm::vec3(
    //         curand_uniform(state),
    //         curand_uniform(state), 
    //         curand_uniform(state)
    //     ) * extent + min;

    //     p2p1 = glm::normalize(glm::vec3(
    //         curand_uniform(state) - 0.5f,
    //         curand_uniform(state) - 0.5f, 
    //         curand_uniform(state) - 0.5f
    //     ));

    //     stack_sizes[i] = 1;
    //     (stack + i * stack_limit)[0] = 0;
    //     hit = bvh_traverse(Ray{p1, p2p1}, dp, st, TraverseMode::CLOSEST_BBOX);
    // } while (!hit.hit);

    // glm::vec3 ray_origin = p1 + hit.t1 * p2p1;
    // glm::vec3 ray_end = p1 + hit.t2 * p2p1;

    // stack_sizes[i] = 1;
    // (stack + i * stack_limit)[0] = 0;
    // HitResult prim_hit = bvh_traverse(Ray{ray_origin, p2p1}, dp, st, TraverseMode::CLOSEST_PRIMITIVE);

    // ray_origins[i] = ray_origin;
    // ray_ends[i] = ray_end;
    // masks[i] = prim_hit.hit;
    // t[i] = prim_hit.t;
}
