#pragma once

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>

#include "cpu_traverse.h"

#define BLOCK_SIZE 256

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states);

CUDA_GLOBAL void traverse_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    HitResults o_hits,
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
    HitResults o_hits,
    int n_rays
);

CUDA_GLOBAL void fill_history_entry(
    bool *i_masks,
    uint32_t *i_node_idxs,
    const BVHDataPointers i_dp,
    DepthInfos o_di,
    int n_rays
);

struct GPUTraverser {
    const BVHData &bvh;

    thrust::device_vector<glm::vec3> vertices;
    thrust::device_vector<Face> faces;
    thrust::device_vector<BVHNode> nodes;
    thrust::device_vector<uint32_t> prim_idxs;

    thrust::device_vector<uint32_t> nbvh_leaf_idxs;
    int n_nbvh_leaves;

    thrust::device_vector<curandState> rand_states;
    int n_rand_states;

    GPUTraverser(const BVHData &bvh) : bvh(bvh) {
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
            grow_nbvh_();
        }
    }

    void grow_nbvh_() {
        std::vector<uint32_t> new_nbvh_leaf_idxs_cpu;
        for (int i = 0; i < nbvh_leaf_idxs.size(); i++) {
            uint32_t leaf_idx = nbvh_leaf_idxs[i];
            BVHNode leaf = nodes[leaf_idx];
            if (leaf.is_leaf()) {
                new_nbvh_leaf_idxs_cpu.push_back(leaf_idx);
            } else {
                leaf.is_nbvh_leaf_ = 0;
                nodes[leaf_idx] = leaf;

                BVHNode left = nodes[leaf.left(leaf_idx)];
                left.is_nbvh_leaf_ = 1;
                nodes[leaf.left(leaf_idx)] = left;

                BVHNode right = nodes[leaf.right()];
                right.is_nbvh_leaf_ = 1;
                nodes[leaf.right()] = right;

                new_nbvh_leaf_idxs_cpu.push_back(leaf.left(leaf_idx));
                new_nbvh_leaf_idxs_cpu.push_back(leaf.right());
            }
        }

        nbvh_leaf_idxs = new_nbvh_leaf_idxs_cpu;
        n_nbvh_leaves = nbvh_leaf_idxs.size();
    }

    BVHDataPointers get_data_pointers() const {
        return {vertices.data().get(), faces.data().get(), nodes.data().get(), prim_idxs.data().get()};
    }

    void init_rand_state(int n_states) {
        this->n_rand_states = n_states;
        rand_states.resize(n_states);
        init_rand_state_entry<<<(n_states + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(rand_states.data().get(), n_states);
    }

    void bbox_raygen(
        glm::vec3 *o_ray_origs,
        glm::vec3 *o_ray_ends,
        bool *o_masks,
        float *o_t1,
        uint32_t *o_node_idxs,
        glm::vec3 *o_normals,
        int *io_stack_sizes,
        uint32_t *io_stack,
        int n_rays
    ) {
        Rays rays = {o_ray_origs, o_ray_ends};
        HitResults o_hits = {o_masks, o_t1, nullptr, o_node_idxs, o_normals};
        StackInfos stack_infos = {64, io_stack_sizes, io_stack};

        bbox_raygen_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            get_data_pointers(),
            stack_infos,
            rand_states.data().get(),
            nbvh_leaf_idxs.data().get(),
            n_nbvh_leaves,
            rays,
            o_hits,
            n_rays
        );
    }

    bool traverse(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_t1,
        float *o_t2,
        uint32_t *o_bbox_idxs,
        glm::vec3 *o_normals,
        int *io_stack_sizes,
        uint32_t *stack,
        int n_rays,
        TreeType tree_type,
        TraverseMode traverse_mode
    ) {
        int *d_alive = nullptr;
        if (traverse_mode == TraverseMode::ANOTHER_BBOX) {
            cudaMalloc(&d_alive, sizeof(int));
            cudaMemset(d_alive, 0, sizeof(int));
        }

        Rays rays = {i_ray_origs, i_ray_vecs};
        HitResults hits = {o_masks, o_t1, o_t2, o_bbox_idxs, o_normals};
        if (traverse_mode == TraverseMode::CLOSEST_PRIMITIVE) {
            hits.t2 = nullptr;
        } else {
            hits.normals = nullptr;
        }
        StackInfos stack_infos = {64, io_stack_sizes, stack};

        traverse_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rays,
            get_data_pointers(),
            stack_infos,
            hits,
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

    void fill_history(
        bool *i_masks,
        uint32_t *i_node_idxs,
        int *o_depths,
        uint32_t *o_history,
        int n_rays
    ) {
        DepthInfos di = {64, o_depths, o_history};

        fill_history_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            i_masks,
            i_node_idxs,
            get_data_pointers(),
            di,
            n_rays
        );
    }
};
