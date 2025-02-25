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
    float *t1,
    float *t2,
    int n_rays
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
    bool *masks,
    float *t, // value in [0, 1]
    int n_rays
);

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

    thrust::device_vector<uint32_t> leaf_idxs;
    int n_leaves;

    thrust::device_vector<curandState> rand_states;
    int n_rand_states;

    GPUTraverser(const BVHData &bvh) : bvh(bvh), stack_limit(bvh.depth * 2) {
        vertices = bvh.vertices;
        faces = bvh.faces;
        nodes = bvh.nodes;
        prim_idxs = bvh.prim_idxs;

        std::vector<uint32_t> leaf_idxs_cpu;
        for (int i = 0; i < nodes.size(); i++) {
            if (bvh.nodes[i].is_leaf()) {
                leaf_idxs_cpu.push_back(i);
            }
        }
        leaf_idxs = leaf_idxs_cpu;
        n_leaves = leaf_idxs.size();
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
            leaf_idxs.data().get(),
            n_leaves,
            ray_origins,
            ray_ends,
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
        float *t1,
        float *t2,
        int n_rays
    ) {
        // needs to be wavefront

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
            t1,
            t2,
            n_rays
        );

        bool alive = thrust::reduce(stack_sizes.begin(), stack_sizes.end()) > 0;
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
