#include "cpu_traverse.h"
#include "utils.h"

#define TRAVERSE_EPS 0.00001f
#define TRAVERSE_STACK_SIZE 64

// at the moment cpu and gpu implementations are the same
// hence this file can be compiled by both nvcc and g++
CUDA_HOST_DEVICE HitResult ray_query(
    const Ray &i_ray,
    const BVHDataPointers &i_dp,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
) {
    uint32_t node_stack[TRAVERSE_STACK_SIZE];
    int cur_stack_size = 0;
    node_stack[cur_stack_size++] = 0; // root node

    HitResult closest_hit = {false, FLT_MAX};

    closest_hit.t = FLT_MAX;

    while (cur_stack_size > 0) {
        uint32_t node_idx = node_stack[--cur_stack_size];
        const BVHNode &node = i_dp.nodes[node_idx];

        bool is_leaf = node.is_leaf();

        if (is_leaf) {
            BBox leaf_bbox = node.bbox;

            HitResult bbox_hit = ray_box_intersection(i_ray, leaf_bbox, allow_negative);
            if (bbox_hit.t > closest_hit.t + TRAVERSE_EPS) {
                continue;
            }

            HitResult node_hit = {false, FLT_MAX};
            for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                const Face &face = i_dp.faces[prim_i];

                HitResult prim_hit = ray_triangle_intersection(i_ray, face, i_dp.vertices, allow_negative);

                glm::vec3 normal = ray_triangle_norm(face, i_dp.vertices);
                float facing = vdot(normal, i_ray.vector);
                if (facing > 0.0f && !allow_backward) {
                    continue;
                }
                if (facing < 0.0f && !allow_forward) {
                    continue;
                }

                if (prim_hit.hit && prim_hit.t < node_hit.t) {
                    prim_hit.prim_idx = prim_i;
                    node_hit = prim_hit;
                }
            }

            if (node_hit.hit && node_hit.t < closest_hit.t) {
                node_hit.prim_idx = node_hit.prim_idx;
                closest_hit = node_hit;
            }
        }

        /* ==== non-leaf case ==== */
        else {
            uint32_t left = node.left();
            uint32_t right = node.right();

            HitResult left_hit = ray_box_intersection(i_ray, i_dp.nodes[left].bbox, allow_negative);
            HitResult right_hit = ray_box_intersection(i_ray, i_dp.nodes[right].bbox, allow_negative);

            if (left_hit.hit && right_hit.hit && (left_hit.t < right_hit.t)) {
                left = left ^ right;
                right = left ^ right;
                left = left ^ right;
            }

            if (left_hit.hit) {
                node_stack[cur_stack_size++] = left;
            }
            
            if (right_hit.hit) {
                node_stack[cur_stack_size++] = right;
            }
        }
    }

    if (!closest_hit.hit) {
        closest_hit.t = 0;
    } else {
        closest_hit.normal = ray_triangle_norm(i_dp.faces[closest_hit.prim_idx], i_dp.vertices);

        if (i_dp.uvs) {
            const Face &face = i_dp.faces[closest_hit.prim_idx];
            float w = 1.0f - closest_hit.bary_u - closest_hit.bary_v;
            closest_hit.uv = w * i_dp.uvs[face.v1]
                           + closest_hit.bary_u * i_dp.uvs[face.v2]
                           + closest_hit.bary_v * i_dp.uvs[face.v3];
        }
    }

    return closest_hit;
}


CUDA_HOST_DEVICE SDFHitResult point_query(
    const glm::vec3 &i_point,
    const BVHDataPointers &i_dp
) {
    uint32_t node_stack[TRAVERSE_STACK_SIZE];
    int cur_stack_size = 0;
    node_stack[cur_stack_size++] = 0; // root node

    SDFHitResult closest_hit = {FLT_MAX};

    while (cur_stack_size > 0) {
        uint32_t node_idx = node_stack[--cur_stack_size];
        const BVHNode &node = i_dp.nodes[node_idx];

        bool is_leaf = node.is_leaf();

        if (is_leaf) {
            float bbox_t = box_df(i_point, node.bbox);
            if (bbox_t > std::abs(closest_hit.t) + TRAVERSE_EPS) {
                continue;
            }

            HitResult node_hit = {false, FLT_MAX};
            for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                const Face &face = i_dp.faces[prim_i];

                SDFHitResult prim_hit = triangle_sdf(i_point, face, i_dp.vertices);
                prim_hit.face_idx = prim_i;             

                if (std::abs(prim_hit.t) < std::abs(closest_hit.t)) {
                    closest_hit = prim_hit;
                }
            }
        }

        /* ==== non-leaf case ==== */
        else {
            uint32_t left = node.left();
            uint32_t right = node.right();

            if (box_df(i_point, i_dp.nodes[left].bbox) < std::abs(closest_hit.t) + TRAVERSE_EPS) {
                node_stack[cur_stack_size++] = left;
            }

            if (box_df(i_point, i_dp.nodes[right].bbox) < std::abs(closest_hit.t) + TRAVERSE_EPS) {
                node_stack[cur_stack_size++] = right;
            }
        }
    }

    return closest_hit;
}
