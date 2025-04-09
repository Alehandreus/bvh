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

    float closest_t = 0;
    if (traverse_mode == TraverseMode::ANOTHER_BBOX) closest_t = o_out.t1[i];

    HitResult hit = bvh_traverse(ray, i_dp, stack_info, traverse_mode, tree_type, true, closest_t);
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

    glm::vec3 p2p1 = glm::vec3(
        curand_uniform(state) - 0.5f,
        curand_uniform(state) - 0.5f, 
        curand_uniform(state) - 0.5f
    );
    if (fabs(p2p1.x) < 1e-6) p2p1.x = 0.0f;
    if (fabs(p2p1.y) < 1e-6) p2p1.y = 0.0f;
    if (fabs(p2p1.z) < 1e-6) p2p1.z = 0.0f;
    if (glm::length(p2p1) < 1e-6) {
        success[i] = 0;
        return;
    }
    p2p1 = glm::normalize(p2p1);

    // glm::vec3 p2p1 = glm::normalize(glm::vec3(
    //     curand_uniform(state) - 0.5f,
    //     curand_uniform(state) - 0.5f, 
    //     curand_uniform(state) - 0.5f
    // ));

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
    // if (hit.hit) hit.normal = ray_triangle_norm(i_dp.faces[hit.prim_idx], i_dp.vertices);
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
    int capacity = 16;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays / capacity) {
        return;
    }

    StackInfo st = io_stack_infos[i];


    /* ==== Generate ray ==== */

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

    glm::vec3 p2p1 = glm::vec3(
        curand_uniform(state) - 0.5f,
        curand_uniform(state) - 0.5f, 
        curand_uniform(state) - 0.5f
    );

    if (fabs(p2p1.x) < 1e-6) p2p1.x = 0.0f;
    if (fabs(p2p1.y) < 1e-6) p2p1.y = 0.0f;
    if (fabs(p2p1.z) < 1e-6) p2p1.z = 0.0f;   
     
    if (glm::length(p2p1) < 1e-6) {
        // success[i] = 0;
        return;
    }

    p2p1 = glm::normalize(p2p1);

    Ray ray = Ray{p1, p2p1};


    /* ==== Obrain true BVH point ==== */

    st.cur_stack_size = 1;
    st.node_stack[0] = 0;
    HitResult bvh_hit = bvh_traverse(ray, i_dp, st, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH, true);
    glm::vec3 hit_point = ray.origin + ray.vector * bvh_hit.t1;


    /* ==== Traverse NBVH ==== */

    int filled = 0;
    HitResult closest_hit = {false, FLT_MAX, 0};

    st.cur_stack_size = 1;
    st.node_stack[0] = 0;

    while (st.cur_stack_size > 0) {
        uint32_t node_idx = st.node_stack[--st.cur_stack_size];
        const BVHNode &node = i_dp.nodes[node_idx];

        bool is_leaf = node.is_leaf() | node.is_nbvh_leaf();
        if (is_leaf) {
            HitResult bbox_hit = ray_box_intersection(ray, node.bbox, true);

            if (!node.bbox.inside(ray.origin + ray.vector * bbox_hit.t1)) {
                continue;
            }

            if (bbox_hit.t1 > closest_hit.t + EPS) {
                continue;
            }

            if (bbox_hit.hit) {
                HitResult res_hit;
                bool inside = node.bbox.inside(hit_point);
                // bool inside = bbox_hit.t1 < bvh_hit.t1 && bvh_hit.t1 < bbox_hit.t2;
                res_hit.hit = bvh_hit.hit && inside;

                if (res_hit.hit) {
                    res_hit.t1 = (bvh_hit.t1 - bbox_hit.t1);
                    // res_hit.t1 = (bvh_hit.t1 - bbox_hit.t1) / (bbox_hit.t2 - bbox_hit.t1);
                } else {
                    res_hit.t1 = 0;
                }

                // TODO: fix this damn shit
                if (res_hit.t1 < 0) {
                    continue;
                }

                res_hit.node_idx = node_idx;

                if (res_hit.hit) {
                    res_hit.normal = bvh_hit.normal;
                } else {
                    res_hit.normal = {0, 0, 0};
                }

                o_hits.fill(i * capacity + filled, res_hit);

                Ray res_points = {ray.origin + ray.vector * bbox_hit.t1, ray.origin + ray.vector * bbox_hit.t2};
                io_rays.fill(i * capacity + filled, res_points);

                success[i * capacity + filled] = 1;
                filled++;
                if (filled >= capacity) break;

                closest_hit = bbox_hit;

                if (res_hit.hit) break;
            }
        } else {
            uint32_t left = node.left();
            uint32_t right = node.right();

            HitResult left_hit = ray_box_intersection(ray, i_dp.nodes[left].bbox, true);
            HitResult right_hit = ray_box_intersection(ray, i_dp.nodes[right].bbox, true);

            if (left_hit.hit && right_hit.hit && (left_hit.t1 < right_hit.t1)) {
                left = left ^ right;
                right = left ^ right;
                left = left ^ right;
            }

            if (left_hit.hit) st.node_stack[st.cur_stack_size++] = left;            
            if (right_hit.hit) st.node_stack[st.cur_stack_size++] = right;
        }
    }

    for (; filled < capacity; ++filled) {
        success[i * capacity + filled] = 0;
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
