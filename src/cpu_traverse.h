#pragma once

#include <glm/glm.hpp>
// #include <omp.h>

#include <fstream>
#include <string>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <tuple>

// don't ask me why these are needed for CPU code
// #include <thrust/host_vector.h>
// #include <thrust/sort.h>
// #include <thrust/execution_policy.h>

#include "build.h"

#define TRAVERSE_EPS 0.00001f
#define TRAVERSE_STACK_SIZE 64

struct BVHDataPointers {
    const glm::vec3 *vertices;
    const Face *faces;
    const BVHNode *nodes;
    const glm::vec2 *uvs;
};

CUDA_HOST_DEVICE HitResult ray_query(
    const Ray &ray,
    const BVHDataPointers &dp,
    bool allow_negative = false
);

CUDA_HOST_DEVICE SDFHitResult point_query(
    const glm::vec3 &i_point,
    const BVHDataPointers &i_dp
);

struct CPUTraverser {
    const BVHData &bvh; 

    CPUTraverser(const BVHData &bvh) : bvh(bvh) {} 

    BVHDataPointers get_data_pointers() const {
        return {bvh.vertices.data(), bvh.faces.data(), bvh.nodes.data(), bvh.uvs.empty() ? nullptr : bvh.uvs.data()};
    }

    HitResult closest_primitive_single(const Ray &ray) const {
        return ::ray_query(ray, get_data_pointers());
    }

    void ray_query(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_t,
        uint32_t *o_prim_idx,
        glm::vec3 *o_normals,
        glm::vec2 *o_uvs,
        int n_rays
    ) {
        Rays rays = {i_ray_origs, i_ray_vecs};
        HitResults hits = {o_masks, o_t, o_prim_idx, o_normals, o_uvs};

        #pragma omp parallel for
        for (int i = 0; i < n_rays; i++) {
            Ray ray = rays[i];

            HitResult hit = ::ray_query(ray, get_data_pointers());
            o_masks[i] = hit.hit;
            if (hit.hit) {
                hits.fill(i, hit);
            }
        }
    }

    SDFHitResult point_query_single(const glm::vec3 &point) const {
        return ::point_query(point, get_data_pointers());
    }

    void point_query(
        glm::vec3 *i_points,
        float *o_t,
        glm::vec3 *o_closests,
        int n_points
    ) {        
        SDFHitResults hits = {o_t, o_closests};

        #pragma omp parallel for
        for (int i = 0; i < n_points; i++) {
            SDFHitResult hit = ::point_query(i_points[i], get_data_pointers());
            hits.fill(i, hit);
        }
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
