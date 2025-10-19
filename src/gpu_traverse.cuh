#pragma once

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <curand_kernel.h>

#include "cpu_traverse.h"

#define BLOCK_SIZE 256

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states);

CUDA_GLOBAL void traverse_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_hits,
    int n_rays
);

struct GPUTraverser {
    const BVHData &bvh;

    thrust::device_vector<glm::vec3> vertices;
    thrust::device_vector<Face> faces;
    thrust::device_vector<BVHNode> nodes;

    thrust::device_vector<uint32_t> node_stack;
    thrust::device_vector<int> cur_stack_sizes;    

    thrust::device_vector<uint32_t> nbvh_leaf_idxs;
    int n_nbvh_leaves;

    GPUTraverser(const BVHData &bvh) : bvh(bvh) {
        vertices = bvh.vertices;
        faces = bvh.faces;
        nodes = bvh.nodes;

        BVHNode root = nodes[0];
        root.is_nbvh_leaf_ = 1;
        nodes[0] = root;
        
        std::vector<uint32_t> a = { 0 };
        nbvh_leaf_idxs = a;
        n_nbvh_leaves = nbvh_leaf_idxs.size();
    }

    void reset_stack(int n_rays) {
        node_stack.resize(n_rays * 64);
        thrust::fill(node_stack.begin(), node_stack.end(), 0);

        cur_stack_sizes.resize(n_rays, 1);
        thrust::fill(cur_stack_sizes.begin(), cur_stack_sizes.end(), 1);
    } 

    BVHDataPointers get_data_pointers() const {
        return {vertices.data().get(), faces.data().get(), nodes.data().get()};
    }

    void traverse(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_t1,
        float *o_t2,
        uint32_t *o_prim_idxs,
        glm::vec3 *o_normals,
        int n_rays
    ) {
        Rays rays = {i_ray_origs, i_ray_vecs};
        HitResults hits = {o_masks, o_t1, o_t2, o_prim_idxs, o_normals};
        hits.t2 = nullptr;

        traverse_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rays,
            get_data_pointers(),
            hits,
            n_rays
        );
    }
};
