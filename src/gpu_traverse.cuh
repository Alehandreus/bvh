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
);

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
);

CUDA_GLOBAL void bbox_raygen_entry_new(
    const BVHDataPointers i_dp,
    StackInfos io_stack_infos,
    curandState *io_rand_states,
    Rays io_rays,
    HitResults o_hits,
    int *success,
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
        node_stack.resize(n_rays * 64);
        thrust::fill(node_stack.begin(), node_stack.end(), 0);

        cur_stack_sizes.resize(n_rays, 1);
        thrust::fill(cur_stack_sizes.begin(), cur_stack_sizes.end(), 1);
    }

    StackInfos get_stack_infos() {
        return {64, cur_stack_sizes.data().get(), node_stack.data().get()};
    }    

    BVHDataPointers get_data_pointers() const {
        return {vertices.data().get(), faces.data().get(), nodes.data().get()};
    }

    bool traverse(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_t1,
        float *o_t2,
        uint32_t *o_bbox_idxs,
        glm::vec3 *o_normals,
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
        HitResults hits = {o_masks, o_t1, o_t2, o_bbox_idxs, o_normals};
        if (traverse_mode == TraverseMode::CLOSEST_PRIMITIVE) {
            hits.t2 = nullptr;
        } else {
            hits.normals = nullptr;
        }
        StackInfos stack_infos = get_stack_infos();

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

struct GPURayGen {
    GPUTraverser &traverser;

    thrust::device_vector<glm::vec3> all_ray_origs;
    thrust::device_vector<glm::vec3> all_ray_ends;
    thrust::device_vector<bool> all_masks;
    thrust::device_vector<float> all_t1;
    thrust::device_vector<uint32_t> all_bbox_idxs;
    thrust::device_vector<glm::vec3> all_normals;

    thrust::device_vector<int> stack_sizes;
    thrust::device_vector<uint32_t> stack;

    thrust::device_vector<int> success;
    thrust::device_vector<int> prefix_sum;

    thrust::device_vector<curandState> rand_states;

    int n_rays;

    GPURayGen(GPUTraverser &traverser, int n_rays) : traverser(traverser), n_rays(n_rays) {
        all_ray_origs.resize(n_rays);
        all_ray_ends.resize(n_rays);
        all_masks.resize(n_rays);
        all_t1.resize(n_rays);
        all_bbox_idxs.resize(n_rays);
        all_normals.resize(n_rays);

        stack_sizes.resize(n_rays, 0);
        stack.resize(n_rays * 64, 0);

        success.resize(n_rays);
        prefix_sum.resize(n_rays);

        rand_states.resize(n_rays);
        init_rand_state_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(rand_states.data().get(), n_rays);
    }

    int raygen(
        glm::vec3 *o_ray_origs,
        glm::vec3 *o_ray_ends,
        bool *o_masks,
        float *o_t1,
        uint32_t *o_bbox_idxs,
        glm::vec3 *o_normals
    ) {
        Rays all_rays = {all_ray_origs.data().get(), all_ray_ends.data().get()};
        HitResults all_hits = {all_masks.data().get(), all_t1.data().get(), nullptr, all_bbox_idxs.data().get(), all_normals.data().get()};
        StackInfos stack_infos = {64, stack_sizes.data().get(), stack.data().get()};

        // bbox_raygen_entry_old<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        //     traverser.get_data_pointers(),
        //     stack_infos,
        //     rand_states.data().get(),
        //     traverser.nbvh_leaf_idxs.data().get(),
        //     traverser.n_nbvh_leaves,
        //     all_rays,
        //     all_hits,
        //     success.data().get(),
        //     n_rays
        // );

        bbox_raygen_entry_new<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            traverser.get_data_pointers(),
            stack_infos,
            rand_states.data().get(),
            all_rays,
            all_hits,
            success.data().get(),
            n_rays
        );

        thrust::exclusive_scan(success.begin(), success.end(), prefix_sum.begin());
        int n_generated = prefix_sum[n_rays - 1] + success[n_rays - 1];

        compact_rays<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            success.data().get(),
            prefix_sum.data().get(),

            all_ray_origs.data().get(),
            all_ray_ends.data().get(),
            all_masks.data().get(),
            all_t1.data().get(),
            nullptr,
            all_bbox_idxs.data().get(),
            all_normals.data().get(),

            o_ray_origs,
            o_ray_ends,
            o_masks,
            o_t1,
            nullptr,
            o_bbox_idxs,
            o_normals,

            n_rays
        );

        return n_generated;
    }
};
