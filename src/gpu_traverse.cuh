#pragma once

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>

#include "cpu_traverse.h"

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
);

CUDA_GLOBAL void another_bbox_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors,
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_limit,
    bool *masks,
    uint32_t *node_idxs,
    uint32_t *nn_idxs,
    float *t1,
    float *t2,
    int *alive,
    int n_rays,
    TreeType tree_type
);

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
);

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
    uint32_t *bbox_idxs,
    uint32_t *nn_idxs,
    bool *masks,
    float *t, // value in [0, 1]
    int n_rays
) ;

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states);

struct GPUTraverser {
    const BVHData &bvh;

    thrust::device_vector<glm::vec3> vertices;
    thrust::device_vector<Face> faces;
    thrust::device_vector<BVHNode> nodes;
    thrust::device_vector<uint32_t> prim_idxs;
    thrust::device_vector<uint32_t> stack;
    thrust::device_vector<int> stack_sizes;
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

    void assign_nns(int node_idx_, int nn, int depth) {
        uint32_t node_idx = node_idx_;
        BVHNode node = nodes[node_idx];
        node.nn = nn;
        nodes[node_idx] = node;

        // cout << node_idx << " " << nn << " " << depth << endl;

        if (node.is_leaf()) {
            return;
        }

        if (depth <= 0) {
            assign_nns(node.left(), nn, depth);
            assign_nns(node.right(), nn, depth);
        } else {
            assign_nns(node.left(), nn * 2, depth - 1);
            assign_nns(node.right(), nn * 2 + 1, depth - 1);
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

        // for (auto i : nbvh_leaf_idxs) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        nbvh_leaf_idxs = new_nbvh_leaf_idxs_cpu;
        n_nbvh_leaves = nbvh_leaf_idxs.size();

        // for (auto i : nbvh_leaf_idxs) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;
    }

    void reset_stack(int n_rays) {
        stack.resize(n_rays * stack_limit);
        thrust::fill(stack.begin(), stack.end(), 0);
        stack_sizes.resize(n_rays, 1);
        thrust::fill(stack_sizes.begin(), stack_sizes.end(), 1);
    }

    BVHDataPointers data_pointers() const {
        return {vertices.data().get(), faces.data().get(), nodes.data().get(), prim_idxs.data().get()};
    }

    void init_rand_state(int n_states) {
        this->n_rand_states = n_states;
        rand_states.resize(n_states);
        init_rand_state_entry<<<(n_states + 31) / 32, 32>>>(rand_states.data().get(), n_states);
    }

    void bbox_raygen(
        glm::vec3 *ray_origins,
        glm::vec3 *ray_ends,
        uint32_t *bbox_idxs,
        uint32_t *nn_idxs,
        bool *masks,
        float *t,
        int n_rays
    ) {
        reset_stack(n_rays);

        bbox_raygen_entry
        <<< (n_rays + 31) / 32, 32 >>>
        (
            data_pointers(),
            stack.data().get(),
            stack_sizes.data().get(),
            stack_limit,
            rand_states.data().get(),
            nbvh_leaf_idxs.data().get(),
            n_nbvh_leaves,
            ray_origins,
            ray_ends,
            bbox_idxs,
            nn_idxs,
            masks,
            t,
            n_rays
        );
    }

    void closest_primitive(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        float *t,
        int n_rays
    ) {
        reset_stack(n_rays);

        closest_primitive_entry
        <<< (n_rays + 31) / 32, 32 >>>
        (
            ray_origins,
            ray_vectors,
            data_pointers(),
            stack.data().get(),
            stack_sizes.data().get(),
            stack_limit,
            masks,
            t,
            n_rays
        );
    }

    bool another_bbox(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        uint32_t *node_idxs,
        uint32_t *nn_idxs,
        float *t1,
        float *t2,
        int n_rays,
        TreeType tree_type
    ) {
        // needs to be wavefront

        int *d_alive;
        cudaMalloc(&d_alive, sizeof(int));
        cudaMemset(d_alive, 0, sizeof(int));
        
        another_bbox_entry
        <<< (n_rays + 31) / 32, 32 >>>
        (
            ray_origins,
            ray_vectors,
            data_pointers(),
            stack.data().get(),
            stack_sizes.data().get(),
            stack_limit,
            masks,
            node_idxs,
            nn_idxs,
            t1,
            t2,
            d_alive,
            n_rays,
            tree_type
        );

        int alive;
        cudaMemcpy(&alive, d_alive, sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_alive);

        return alive;
    }

    void segments(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *segments,
        int n_rays,
        int n_segments
    ) {
        reset_stack(n_rays);
        
        cudaMemset(segments, 0, n_rays * n_segments * sizeof(bool));

        segments_entry
        <<< (n_rays + 31) / 32, 32 >>>
        (
            ray_origins,
            ray_vectors,
            data_pointers(),
            stack.data().get(),
            stack_sizes.data().get(),
            stack_limit,
            segments,
            n_rays,
            n_segments
        );
    }
};
