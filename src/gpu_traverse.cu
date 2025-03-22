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

    HitResult hit = bvh_traverse(ray, i_dp, stack_info, traverse_mode, tree_type);
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
    Rays o_rays,
    HitResults o_hits,
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
        curand_uniform(state) * 0.98 + 0.01,
        curand_uniform(state) * 0.98 + 0.01, 
        curand_uniform(state) * 0.98 + 0.01
    ) * extent + min;

    glm::vec3 p2p1 = glm::normalize(glm::vec3(
        curand_uniform(state) - 0.5f,
        curand_uniform(state) - 0.5f, 
        curand_uniform(state) - 0.5f
    ));

    HitResult hit = ray_box_intersection(Ray{p1, p2p1}, leaf.bbox);
    assert(hit.hit);
    
    glm::vec3 ray_origin = p1 + hit.t1 * p2p1;
    glm::vec3 ray_end = p1 + hit.t2 * p2p1;
    glm::vec3 ray_vector = p2p1;


    // ==== Intersect the primitives ==== //

    Ray ray = Ray{ray_origin, ray_vector};
    StackInfo st = io_stack_infos[i];
    st.node_stack[0] = leaf_idx;
    hit = bvh_traverse(ray, i_dp, st, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
    hit.node_idx = leaf_idx;
    o_hits.fill(i, hit);
    o_rays.fill(i, {ray_origin, ray_end});
}

CUDA_GLOBAL void fill_history_entry(
    bool *i_mask,
    uint32_t *i_node_idxs,
    const BVHDataPointers i_dp,
    DepthInfos o_di,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays || !i_mask[i]) {
        return;
    }

    int depth = 0;

    uint32_t node_idx = i_node_idxs[i];
    BVHNode node = i_dp.nodes[node_idx];
    o_di.bbox_idxs[i * 64 + depth++] = node_idx;

    while (node_idx != 0) {
        node_idx = node.father;
        node = i_dp.nodes[node_idx];
        o_di.bbox_idxs[i * 64 + depth++] = node_idx;
    }

    for (int j = 0; j < depth / 2; j++) {
        uint32_t tmp = o_di.bbox_idxs[i * 64 + j];
        o_di.bbox_idxs[i * 64 + j] = o_di.bbox_idxs[i * 64 + depth - j - 1];
        o_di.bbox_idxs[i * 64 + depth - j - 1] = tmp;
    }

    o_di.cur_depths[i] = depth;
}
