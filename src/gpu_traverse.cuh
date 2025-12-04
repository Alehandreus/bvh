#pragma once

#include <thrust/device_vector.h>
#include <curand_kernel.h>

#include "cpu_traverse.h"

#define BLOCK_SIZE 256

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states);

CUDA_GLOBAL void ray_query_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_hits,
    int n_rays
);

CUDA_GLOBAL void point_query_entry(
    const glm::vec3 *i_points,
    const BVHDataPointers i_dp,
    SDFHitResults o_out,
    int n_points
);

CUDA_GLOBAL void ray_query_all_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_hits,
    uint32_t *o_n_hits,
    int max_hits_per_ray,
    int n_rays
);

struct GPUTraverser {
    const BVHData &bvh;

    thrust::device_vector<glm::vec3> vertices;
    thrust::device_vector<Face> faces;
    thrust::device_vector<BVHNode> nodes;

    GPUTraverser(const BVHData &bvh) : bvh(bvh) {
        vertices = bvh.vertices;
        faces = bvh.faces;
        nodes = bvh.nodes;
    }

    BVHDataPointers get_data_pointers() const {
        return {vertices.data().get(), faces.data().get(), nodes.data().get()};
    }

    void ray_query(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_dists,
        uint32_t *o_prim_idxs,
        glm::vec3 *o_normals,
        int n_rays
    ) {
        Rays rays = {i_ray_origs, i_ray_vecs};
        HitResults hits = {o_masks, o_dists, o_prim_idxs, o_normals};

        ray_query_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rays,
            get_data_pointers(),
            hits,
            n_rays
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in ray_query: %s\n", cudaGetErrorString(err));
        }
    }

    void ray_query_all(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_dists,
        uint32_t *o_prim_idxs,
        uint32_t *o_n_hits,
        int max_hits_per_ray,
        int n_rays
    ) {
        Rays rays = {i_ray_origs, i_ray_vecs};
        HitResults hits = {o_masks, o_dists, o_prim_idxs, nullptr};

        ray_query_all_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rays,
            get_data_pointers(),
            hits,
            o_n_hits,
            max_hits_per_ray,
            n_rays
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in ray_query_all: %s\n", cudaGetErrorString(err));
        }
    }    

    void point_query(
        const glm::vec3 *i_points,
        float *o_t,
        glm::vec3 *o_closests,
        glm::vec3 *o_barycentricses,
        uint32_t *o_face_idxs,
        int n_points
    ) {
        SDFHitResults out = {o_t, o_closests, o_barycentricses, o_face_idxs};

        point_query_entry<<<(n_points + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            i_points,
            get_data_pointers(),
            out,
            n_points
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in point_query: %s\n", cudaGetErrorString(err));
        }
    }
};
