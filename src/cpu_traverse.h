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

enum class TreeType {
    BVH,
    NBVH
};

struct BVHDataPointers {
    const glm::vec3 *vertices;
    const Face *faces;
    const BVHNode *nodes;
    const uint32_t *prim_idxs;
};

struct StackInfo {
    int &cur_stack_size;
    uint32_t *node_stack;
    int *depth_stack;
};

struct StackInfos {
    int stack_limit;
    int *cur_stack_sizes;
    uint32_t *node_stacks;
    int *depth_stacks;

    CUDA_HOST_DEVICE StackInfo operator[](int i) {
        return {
            cur_stack_sizes[i],
            node_stacks + i * stack_limit,
            depth_stacks + i * stack_limit
        };
    }
};

struct DepthInfo {
    int &cur_depth;
    uint32_t *bbox_idxs;
};

struct DepthInfos {
    int depth_limit;
    int *cur_depths;
    uint32_t *bbox_idxs;

    CUDA_HOST_DEVICE DepthInfo operator[](int i) {
        return {
            cur_depths[i],
            bbox_idxs + i * depth_limit
        };
    }
};

enum class TraverseMode {
    CLOSEST_PRIMITIVE,
    CLOSEST_BBOX,
    ANOTHER_BBOX    
};

CUDA_HOST_DEVICE HitResult bvh_traverse(
    const Ray &ray,
    const BVHDataPointers &dp,
    StackInfo &st,
    DepthInfo &di,
    TraverseMode mode,
    TreeType tree_type
);

struct CPUTraverser {
    const BVHData &bvh;

    std::vector<uint32_t> stack;
    std::vector<int> cur_stack_sizes;
    std::vector<uint32_t> bbox_idxs;
    std::vector<int> cur_depths;
    int stack_limit;

    CPUTraverser(const BVHData &bvh) : bvh(bvh), stack_limit(bvh.depth * 2) {}

    void reset_stack(int n_rays) {
        stack.resize(n_rays * stack_limit);
        std::fill(stack.begin(), stack.end(), 0);

        cur_stack_sizes.resize(n_rays, 1);
        std::fill(cur_stack_sizes.begin(), cur_stack_sizes.end(), 1);

        bbox_idxs.resize(n_rays * bvh.depth);
        std::fill(bbox_idxs.begin(), bbox_idxs.end(), 0);

        cur_depths.resize(n_rays, 0);
        std::fill(cur_depths.begin(), cur_depths.end(), 0);
    }

    BVHDataPointers get_data_pointers() const {
        return {bvh.vertices.data(), bvh.faces.data(), bvh.nodes.data(), bvh.prim_idxs.data()};
    }

    StackInfos get_stack_infos() {
        return {stack_limit, cur_stack_sizes.data(), stack.data()};
    }

    // traverse single ray, use local stack to be thread-safe
    HitResult closest_primitive_single(const Ray &ray) const {
        std::vector<uint32_t> smol_node_stack(stack_limit, 0);
        std::vector<int> smol_depth_stack(stack_limit, 0);
        int smol_stack_size = 1;

        std::vector<uint32_t> smol_bbox_idxs(bvh.depth, 0);
        int smol_depth = 0;

        StackInfo stack_info = {smol_stack_size, smol_node_stack.data(), smol_depth_stack.data()};
        DepthInfo depth_info = {smol_depth, smol_bbox_idxs.data()};

        return bvh_traverse(ray, get_data_pointers(), stack_info, depth_info, TraverseMode::CLOSEST_PRIMITIVE, TreeType::BVH);
    }

    bool traverse(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_t1,
        float *o_t2,
        int *io_depths,
        uint32_t *io_bbox_idxs,
        int n_rays,
        TreeType tree_type,
        TraverseMode traverse_mode
    ) {
        if (traverse_mode != TraverseMode::ANOTHER_BBOX) {
            reset_stack(n_rays);
        }

        int alive = false;

        Rays rays = {i_ray_origs, i_ray_vecs};
        StackInfos stack_infos = get_stack_infos();

        #pragma omp parallel for reduction(||: alive)
        for (int i = 0; i < n_rays; i++) {
            Ray ray = rays[i];
            StackInfo stack_info = stack_infos[i];
            DepthInfo depth_info = {io_depths[i], io_bbox_idxs + i * bvh.depth};

            HitResult hit = bvh_traverse(ray, get_data_pointers(), stack_info, depth_info, traverse_mode, TreeType::BVH);
            o_masks[i] = hit.hit;
            if (hit.hit) {
                o_t1[i] = hit.t1;
                o_t2[i] = hit.t2;
            }

            alive = alive || hit.hit;
        }

        return alive;
    }

    void generate_camera_rays(
        glm::vec3 *o_ray_origs,
        glm::vec3 *o_ray_vecs,
        bool *o_masks,
        float *o_t,
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

                o_ray_origs[y * img_size + x] = cam_pos;
                o_ray_vecs[y * img_size + x] = dir;

                o_masks[y * img_size + x] = hit.hit;
                o_t[y * img_size + x] = hit.t;
            }
        }        
    }
};
