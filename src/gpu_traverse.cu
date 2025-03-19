#include <curand_kernel.h>

#include "gpu_traverse.cuh"

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
    StackInfos io_stack_infos,
    HitResults o_out,
    DepthInfos io_depth_infos,
    int n_rays,
    TreeType tree_type,
    TraverseMode traverse_mode,
    int *o_alive
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];
    StackInfo stack_info = io_stack_infos[i];
    DepthInfo depth_info = io_depth_infos[i];

    HitResult hit = bvh_traverse(ray, i_dp, stack_info, depth_info, traverse_mode, tree_type);
    o_out.fill(i, hit);

    if (traverse_mode == TraverseMode::ANOTHER_BBOX) {
        // I am sorry
        atomicOr(o_alive, hit.hit);
    }
}

CUDA_GLOBAL void bbox_raygen_entry(
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    curandState *io_rand_states,
    uint32_t *i_leaf_idxs,
    int n_leaves,
    DepthInfos io_depth_infos,
    Rays o_rays,
    HitResults o_out,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }


    // ==== Generate random leaf index ==== //

    curandState *state = &io_rand_states[i];

    uint32_t leaf_idx = i_leaf_idxs[curand(state) % n_leaves];
    const BVHNode &leaf = i_dp.nodes[leaf_idx];


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
        o_rays[i] = Ray{p1 * 0.f, p1 * 0.f};
        o_out.fill(i, bbox_hit);
        return;
    }
    
    glm::vec3 ray_origin = p1 + bbox_hit.t1 * p2p1;
    glm::vec3 ray_end = p1 + bbox_hit.t2 * p2p1;
    glm::vec3 ray_vector = p2p1;


    // ==== Intersect the primitives ==== //

    Ray ray = Ray{ray_origin, ray_vector};
    StackInfo st = io_stack_infos[i];
    DepthInfo depth_info = io_depth_infos[i];
    HitResult hit = bvh_traverse(ray, i_dp, st, depth_info, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
    o_out.fill(i, hit);
    o_rays[i] = ray;
}
