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

CUDA_GLOBAL void bbox_raygen_entry_old(
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    curandState *io_rand_states,
    uint32_t *i_leaf_idxs,
    int n_leaves,    
    Rays io_rays,
    HitResults o_hits,
    int *success,
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
    if (!hit.hit) {
        success[i] = 0;
        return;
    }
    
    glm::vec3 ray_origin = p1 + hit.t1 * p2p1;
    glm::vec3 ray_end = p1 + hit.t2 * p2p1;
    glm::vec3 ray_vector = p2p1;


    // ==== Intersect the primitives ==== //

    Ray ray = Ray{ray_origin, ray_vector};
    StackInfo st = io_stack_infos[i];
    st.cur_stack_size = 1;
    st.node_stack[0] = leaf_idx;
    hit = bvh_traverse(ray, i_dp, st, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
    hit.node_idx = leaf_idx;
    if (hit.hit) hit.normal = ray_triangle_norm(i_dp.faces[hit.prim_idx], i_dp.vertices);
    o_hits.fill(i, hit);
    io_rays.fill(i, {ray_origin, ray_end});
    
    success[i] = 1;
}

CUDA_GLOBAL void bbox_raygen_entry_new(
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    curandState *io_rand_states,
    Rays io_rays,
    HitResults o_hits,
    int *success,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }


    /* ==== Generate ray ==== */

    StackInfo st = io_stack_infos[i];

    if (st.cur_stack_size == 0) {
        curandState *state = &io_rand_states[i];

        BBox outer_bbox = i_dp.nodes[0].bbox;
        glm::vec3 extent = outer_bbox.diagonal();
        outer_bbox.min -= extent * 0.5f;
        outer_bbox.max += extent * 0.5f;

        glm::vec3 p1 = glm::vec3(
            curand_uniform(state) * 0.90 + 0.05,
            curand_uniform(state) * 0.90 + 0.05, 
            curand_uniform(state) * 0.90 + 0.05
        ) * outer_bbox.diagonal() + outer_bbox.min;

        glm::vec3 p2p1 = glm::normalize(glm::vec3(
            curand_uniform(state) - 0.5f,
            curand_uniform(state) - 0.5f, 
            curand_uniform(state) - 0.5f
        ));

        io_rays.fill(i, {p1, p1 + p2p1}); // we store ray ends here

        st.cur_stack_size = 1;
        st.node_stack[0] = 0;
    }    


    /* ==== Intersect NBVH ==== */

    Ray points = io_rays[i];
    Ray ray = Ray{points.origin, points.vector - points.origin};
    HitResult nbvh_hit = bvh_traverse(ray, i_dp, st, TraverseMode::ANOTHER_BBOX, TreeType::NBVH);
    success[i] = nbvh_hit.hit;
    if (!nbvh_hit.hit) {
        return;
    }

    points.origin = ray.origin + ray.vector * nbvh_hit.t1;
    points.vector = ray.origin + ray.vector * nbvh_hit.t2;
    io_rays.fill(i, points);
    ray = Ray{points.origin, points.vector - points.origin};


    /* ==== If hit, intersect BVH ==== */

    int bvh_stack_size = 1;
    st.node_stack[st.cur_stack_size] = nbvh_hit.node_idx;
    StackInfo bvh_st = {bvh_stack_size, st.node_stack + st.cur_stack_size};

    HitResult bvh_hit = bvh_traverse(ray, i_dp, bvh_st, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
    if (bvh_hit.hit) bvh_hit.normal = ray_triangle_norm(i_dp.faces[bvh_hit.prim_idx], i_dp.vertices);
    bvh_hit.node_idx = nbvh_hit.node_idx;
    o_hits.fill(i, bvh_hit);

    if (bvh_hit.t1 > 1.01) {
        success[i] = 0;
    }
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

    o_di.cur_depths[i] = depth;
}

CUDA_GLOBAL void compact_rays(
    int *i_success,
    int *i_prefix_sum,

    glm::vec3 *i_ray_origs,
    glm::vec3 *i_ray_ends,
    bool *i_masks,
    float *i_t1,
    float *i_t2,
    uint32_t *i_bbox_idxs,
    glm::vec3 *i_normals,

    glm::vec3 *o_ray_origs,
    glm::vec3 *o_ray_ends,
    bool *io_masks,
    float *io_t1,
    float *io_t2,
    uint32_t *io_bbox_idxs,
    glm::vec3 *io_normals,

    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    if (i_success[i] == 0) {
        return;
    }

    int idx = i_prefix_sum[i];
    o_ray_origs[idx] = i_ray_origs[i];
    o_ray_ends[idx] = i_ray_ends[i];
    io_masks[idx] = i_masks[i];
    io_t1[idx] = i_t1[i];
    if (i_t2) {
        io_t2[idx] = i_t2[i];
    }
    io_bbox_idxs[idx] = i_bbox_idxs[i];
    if (i_normals) {
        io_normals[idx] = i_normals[i];
    }
}
