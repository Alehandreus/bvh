#pragma once

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>

#include "cpu_traverse.h"

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states);

CUDA_GLOBAL void traverse_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    BboxOut o_out,
    int n_rays,
    TreeType tree_type,
    TraverseMode traverse_mode,
    int *o_alive
);

CUDA_GLOBAL void bbox_raygen_entry(
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    curandState *io_rand_states,
    uint32_t *i_leaf_idxs,
    int n_leaves,
    Rays o_rays,
    PrimOut o_out,
    int n_rays
);

struct GPUTraverser {
    const BVHData &bvh;

    thrust::device_vector<glm::vec3> vertices;
    thrust::device_vector<Face> faces;
    thrust::device_vector<BVHNode> nodes;
    thrust::device_vector<uint32_t> prim_idxs;

    thrust::device_vector<uint32_t> stack;
    thrust::device_vector<int> cur_stack_sizes;
    thrust::device_vector<uint32_t> bbox_idxs;
    thrust::device_vector<int> cur_depths;
    int stack_limit;

    thrust::device_vector<uint32_t> nbvh_leaf_idxs;
    int n_nbvh_leaves;

    thrust::device_vector<curandState> rand_states;
    int n_rand_states;

    GPUTraverser(const BVHData &bvh) : bvh(bvh), stack_limit(bvh.depth * 2) {
        vertices = bvh.vertices;
        faces = bvh.faces;
        nodes = bvh.nodes;
        prim_idxs = bvh.prim_idxs;

        BVHNode root = nodes[0];
        root.is_nbvh_leaf_ = 1;
        nodes[0] = root;
        
        // nbvh_leaf_idxs = { 0 };
        std::vector<uint32_t> a = { 0 };
        nbvh_leaf_idxs = a;
        n_nbvh_leaves = nbvh_leaf_idxs.size();
    }

    void grow_nbvh(int steps) {
        for (int i = 0; i < steps; i++) {
            grow_nbvh();
        }
    }

    void grow_nbvh() {
        std::vector<uint32_t> new_nbvh_leaf_idxs_cpu;
        for (int i = 0; i < nbvh_leaf_idxs.size(); i++) {
            uint32_t leaf_idx = nbvh_leaf_idxs[i];
            BVHNode leaf = nodes[leaf_idx];
            if (leaf.is_leaf()) {
                new_nbvh_leaf_idxs_cpu.push_back(leaf_idx);
            } else {
                leaf.is_nbvh_leaf_ = 0;
                nodes[leaf_idx] = leaf;

                BVHNode left = nodes[leaf.left()];
                left.is_nbvh_leaf_ = 1;
                nodes[leaf.left()] = left;

                BVHNode right = nodes[leaf.right()];
                right.is_nbvh_leaf_ = 1;
                nodes[leaf.right()] = right;

                new_nbvh_leaf_idxs_cpu.push_back(leaf.left());
                new_nbvh_leaf_idxs_cpu.push_back(leaf.right());
            }
        }

        nbvh_leaf_idxs = new_nbvh_leaf_idxs_cpu;
        n_nbvh_leaves = nbvh_leaf_idxs.size();
    }

    void reset_stack(int n_rays) {
        stack.resize(n_rays * stack_limit);
        thrust::fill(stack.begin(), stack.end(), 0);

        cur_stack_sizes.resize(n_rays, 1);
        thrust::fill(cur_stack_sizes.begin(), cur_stack_sizes.end(), 1);

        bbox_idxs.resize(n_rays * stack_limit);
        thrust::fill(bbox_idxs.begin(), bbox_idxs.end(), 0);
        
        cur_depths.resize(n_rays, 0);
        thrust::fill(cur_depths.begin(), cur_depths.end(), 0);
    }

    StackInfos get_stack_infos() {
        return {
            stack_limit,
            cur_stack_sizes.data().get(),
            stack.data().get(), 
            
            stack_limit,
            cur_depths.data().get(),
            bbox_idxs.data().get()
        };
    }

    BVHDataPointers get_data_pointers() const {
        return {vertices.data().get(), faces.data().get(), nodes.data().get(), prim_idxs.data().get()};
    }

    void init_rand_state(int n_states) {
        this->n_rand_states = n_states;
        rand_states.resize(n_states);
        init_rand_state_entry<<<(n_states + 31) / 32, 32>>>(rand_states.data().get(), n_states);
    }

    void bbox_raygen(
        glm::vec3 *o_ray_origs,
        glm::vec3 *o_ray_ends,
        uint32_t *o_bbox_idxs,
        bool *o_masks,
        float *o_t,
        int n_rays
    ) {
        reset_stack(n_rays);

        Rays rays = {o_ray_origs, o_ray_ends};
        PrimOut prim_out = {o_masks, o_t};

        bbox_raygen_entry<<<(n_rays + 31) / 32, 32>>>(
            get_data_pointers(),
            get_stack_infos(),
            rand_states.data().get(),
            nbvh_leaf_idxs.data().get(),
            n_nbvh_leaves,
            rays,
            prim_out,
            n_rays
        );
    }

    bool traverse(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        uint32_t *o_bbox_idxs,
        bool *o_masks,
        float *o_t1,
        float *o_t2,
        int n_rays,
        TreeType tree_type,
        TraverseMode traverse_mode
    ) {
        if (traverse_mode != TraverseMode::ANOTHER_BBOX) {
            reset_stack(n_rays);
        }

        int *d_alive = nullptr;
        if (traverse_mode == TraverseMode::ANOTHER_BBOX) {
            cudaMalloc(&d_alive, sizeof(int));
            cudaMemset(d_alive, 0, sizeof(int));
        }

        Rays rays = {i_ray_origs, i_ray_vecs};
        BboxOut bbox_out = {o_masks, o_t1, o_t2};

        traverse_entry<<<(n_rays + 31) / 32, 32>>>(
            rays,
            get_data_pointers(),
            get_stack_infos(),
            bbox_out,
            n_rays,
            tree_type,
            traverse_mode,
            d_alive
        );

        if (traverse_mode == TraverseMode::ANOTHER_BBOX) {
            int alive;
            cudaMemcpy(&alive, d_alive, sizeof(int), cudaMemcpyDeviceToHost);            
            cudaFree(d_alive);
            return alive;
        }

        return false;
    }
};
