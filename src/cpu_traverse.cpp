#include "cpu_traverse.h"

// at the moment cpu and gpu implementations are the same
// hence this file can be compiled by both nvcc and g++
CUDA_HOST_DEVICE HitResult bvh_traverse(
    const Ray &i_ray,
    const BVHDataPointers &i_dp,
    StackInfo &io_st,
    TraverseMode mode,
    TreeType tree_type,
    bool allow_negative,
    float closest_t
) {
    HitResult closest_hit = {false, FLT_MAX, 0};

    if (closest_t != 0) {
        closest_hit.t1 = closest_t;
    }

    constexpr float eps = 0.00001f;

    while (io_st.cur_stack_size > 0) {
        uint32_t node_idx = io_st.node_stack[--io_st.cur_stack_size];
        const BVHNode &node = i_dp.nodes[node_idx];

        bool is_leaf = node.is_leaf(); 
        if (tree_type == TreeType::NBVH) {
            is_leaf |= node.is_nbvh_leaf();
        }

        if (is_leaf) {
            /* ==== intersect only node bbox ==== */
            if (mode == TraverseMode::ANOTHER_BBOX) {
                HitResult bbox_hit = ray_box_intersection(i_ray, node.bbox, allow_negative);
                if (bbox_hit.t1 > closest_hit.t1 + eps) {
                    continue;
                }
                if (bbox_hit.hit) {
                    bbox_hit.node_idx = node_idx;
                    return bbox_hit;
                }
            }

            /* ==== intersect primitives in node ==== */
            else if (mode == TraverseMode::CLOSEST_PRIMITIVE) {
                HitResult bbox_hit = ray_box_intersection(i_ray, node.bbox, allow_negative);
                if (bbox_hit.t1 > closest_hit.t + eps) {
                    continue;
                }

                HitResult node_hit = {false, FLT_MAX};
                for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                    const Face &face = i_dp.faces[prim_i];

                    HitResult prim_hit = ray_triangle_intersection(i_ray, face, i_dp.vertices, allow_negative);

                    if (prim_hit.hit && prim_hit.t < node_hit.t) {
                        prim_hit.prim_idx = prim_i;
                        node_hit = prim_hit;
                    }
                }

                if (node_hit.hit && node_hit.t < closest_hit.t) {
                    node_hit.node_idx = node_idx;
                    closest_hit = node_hit;
                }
            }

            /* ==== idk who needs this but return the closest bbox ==== */
            else if (mode == TraverseMode::CLOSEST_BBOX) {
                HitResult bbox_hit = ray_box_intersection(i_ray, node.bbox, allow_negative);
                bbox_hit.node_idx = node_idx;
                if (bbox_hit.hit && bbox_hit.t1 < closest_hit.t1) {
                    closest_hit = bbox_hit;
                }
            }
        }

        /* ==== non-leaf case ==== */
        else {
            uint32_t left = node.left();
            uint32_t right = node.right();

            HitResult left_hit = ray_box_intersection(i_ray, i_dp.nodes[left].bbox, allow_negative);
            HitResult right_hit = ray_box_intersection(i_ray, i_dp.nodes[right].bbox, allow_negative);

            if (left_hit.hit && right_hit.hit && (left_hit.t1 < right_hit.t1)) {
                left = left ^ right;
                right = left ^ right;
                left = left ^ right;
            }

            if (left_hit.hit) {
                io_st.node_stack[io_st.cur_stack_size++] = left;
            }
            
            if (right_hit.hit) {
                io_st.node_stack[io_st.cur_stack_size++] = right;
            }
        }
    }

    if (!closest_hit.hit) {
        closest_hit.t = 0;
    } else if (mode == TraverseMode::CLOSEST_PRIMITIVE) {
        closest_hit.normal = ray_triangle_norm(i_dp.faces[closest_hit.prim_idx], i_dp.vertices);
    }

    return closest_hit;
}