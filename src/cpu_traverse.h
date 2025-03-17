#pragma once

#include <glm/glm.hpp>
// #include <omp.h>

#include <fstream>
#include <string>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <tuple>

#include "build.h"

struct BVHDataPointers {
    const glm::vec3 *vertices;
    const Face *faces;
    const BVHNode *nodes;
    const uint32_t *prim_idxs;
};

struct StackInfo {
    int &stack_size;
    uint32_t *stack;
};

enum class TraverseMode {
    CLOSEST_PRIMITIVE,
    CLOSEST_BBOX,
    ANOTHER_BBOX    
};

enum class TreeType {
    BVH,
    NBVH
};

CUDA_HOST_DEVICE HitResult bvh_traverse(
    const Ray &ray,
    const BVHDataPointers &dp,
    StackInfo &st,
    TraverseMode mode,
    TreeType tree_type
);

struct CPUTraverser {
    const BVHData &bvh;

    std::vector<uint32_t> stack;
    std::vector<int> stack_sizes;
    int stack_limit;

    CPUTraverser(const BVHData &bvh) : bvh(bvh), stack_limit(bvh.depth * 2) {}

    void reset_stack(int n_rays) {
        stack.resize(n_rays * stack_limit);
        std::fill(stack.begin(), stack.end(), 0);
        stack_sizes.resize(n_rays, 1);
        std::fill(stack_sizes.begin(), stack_sizes.end(), 1);
    }

    BVHDataPointers data_pointers() const {
        return {bvh.vertices.data(), bvh.faces.data(), bvh.nodes.data(), bvh.prim_idxs.data()};
    }

    // traverse single ray, use local stack to be thread-safe
    HitResult closest_primitive_single(const Ray &ray) const {
        std::vector<uint32_t> smol_stack(stack_limit, 0);
        int smol_stack_size = 1;
        StackInfo stack_info = {smol_stack_size, smol_stack.data()};

        return bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
    }

    // this and others are intended for python use, so structures like Ray and HitResult are not exposed
    void closest_primitive(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        float *t,
        int n_rays
    ) {
        reset_stack(n_rays);

        #pragma omp parallel for
        for (int i = 0; i < n_rays; i++) {
            Ray ray = {ray_origins[i], ray_vectors[i]};
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * stack_limit};

            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
            masks[i] = hit.hit;
            t[i] = hit.t;
        }
    }

    void closest_bbox(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        uint32_t *node_idxs,
        float *t1,
        float *t2,
        int n_rays
    ) {
        reset_stack(n_rays);

        #pragma omp parallel for
        for (int i = 0; i < n_rays; i++) {
            Ray ray = {ray_origins[i], ray_vectors[i]};
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * stack_limit};

            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_BBOX, TreeType::BVH);
            masks[i] = hit.hit;
            node_idxs[i] = hit.node_idx;
            t1[i] = hit.t1;
            t2[i] = hit.t2;
        }
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
        bool alive = false;

        #pragma omp parallel for reduction(||: alive)
        for (int i = 0; i < n_rays; i++) {
            Ray ray = {ray_origins[i], ray_vectors[i]};
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * stack_limit};

            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::ANOTHER_BBOX, TreeType::BVH);
            masks[i] = hit.hit;
            node_idxs[i] = hit.node_idx;
            t1[i] = hit.t1;
            t2[i] = hit.t2;

            alive = alive || hit.hit;
        }

        return alive;
    }

    void generate_camera_rays(
        glm::vec3 *ray_origins,
        glm::vec3 *ray_vectors,
        bool *masks,
        float *t,
        int img_size
    ) {
        // ==== Set up default Camera ==== //

        auto [min, max] = bvh.nodes[0].bbox;
        glm::vec3 center = (max + min) * 0.5f;
        float max_extent = std::fmax(max.x - min.x, std::fmax(max.y - min.y, max.z - min.z));
        glm::vec3 cam_pos = { 
            center.x + max_extent * 1.0,
            center.y - max_extent * 1.5,
            center.z + max_extent * 0.5
        };
        glm::vec3 cam_dir = (center - cam_pos) * 0.9f;
        glm::vec3 x_dir = glm::normalize(glm::cross(cam_dir, glm::vec3(0, 0, 1))) * (max_extent / 2);
        glm::vec3 y_dir = -glm::normalize(glm::cross(x_dir, cam_dir)) * (max_extent / 2);


        // ==== Generate Camera Rays ==== //

        #pragma omp parallel for
        for (int y = 0; y < img_size; y++) {
            for (int x = 0; x < img_size; x++) {
                float x_f = ((float)x / img_size - 0.5f) * 2;
                float y_f = ((float)y / img_size - 0.5f) * 2;

                glm::vec3 dir = cam_dir + x_dir * x_f + y_dir * y_f;
                HitResult hit = closest_primitive_single({cam_pos, dir});

                masks[y * img_size + x] = hit.hit;
                t[y * img_size + x] = hit.t;
            }
        }        
    }
};
