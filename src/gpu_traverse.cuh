#include <thrust/device_vector.h>

#include "cpu_traverse.h"

CUDA_GLOBAL void closest_primitive_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors,
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_reserve,
    bool *masks,
    float *t,
    int n_rays
);

CUDA_GLOBAL void segments_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors, // ray_origins and ray_origins + ray_vectors are segment edges
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_reserve,
    bool *segments,
    int n_rays,
    int n_segments
);

struct GPUTraverser {
    const BVHData &bvh;

    thrust::device_vector<glm::vec3> vertices;
    thrust::device_vector<Face> faces;
    thrust::device_vector<BVHNode> nodes;
    thrust::device_vector<uint32_t> prim_idxs;
    thrust::device_vector<uint32_t> stack;
    thrust::device_vector<int> stack_sizes;
    int stack_limit;

    GPUTraverser(const BVHData &bvh) : bvh(bvh), stack_limit(bvh.depth * 2) {
        vertices = bvh.vertices;
        faces = bvh.faces;
        nodes = bvh.nodes;
        prim_idxs = bvh.prim_idxs;
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
