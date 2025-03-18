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
    const Rays i_rays,
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    PrimOut o_out,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];
    StackInfo stack_info = io_stack_infos[i];

    HitResult hit = bvh_traverse(ray, i_dp, stack_info, TraverseMode::CLOSEST_PRIMITIVE, false);
    o_out.fill(i, hit);
}

CUDA_GLOBAL void another_bbox_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    BboxOut o_out,
    int *o_alive,
    int n_rays,
    bool nbvh_only
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];
    StackInfo stack_info = io_stack_infos[i];

    HitResult hit = bvh_traverse(ray, i_dp, stack_info, TraverseMode::ANOTHER_BBOX, nbvh_only);
    o_out.fill(i, hit);

    // I am sorry
    atomicOr(o_alive, hit.hit);
}

CUDA_GLOBAL void bbox_raygen_entry(
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    curandState *io_rand_states,
    uint32_t *i_leaf_idxs,
    int n_leaves,
    Rays o_rays,
    PrimOut o_out,
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

    StackInfo st = io_stack_infos[i];
    HitResult hit = bvh_traverse(Ray{ray_origin, ray_vector}, i_dp, st, TraverseMode::CLOSEST_PRIMITIVE, false);
    o_out.fill(i, hit);
    o_rays[i] = Ray{ray_origin, ray_end};
}
