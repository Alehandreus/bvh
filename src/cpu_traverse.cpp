#include "cpu_traverse.h"

// at the moment cpu and gpu implementations are the same
// hence this file can be compiled by both nvcc and g++
CUDA_HOST_DEVICE HitResult bvh_traverse(
    const Ray &i_ray,
    const BVHDataPointers &i_dp,
    StackInfo &io_st,
    TraverseMode mode,
    TreeType tree_type
) {
    HitResult closest_hit = {false, FLT_MAX, 0};

    while (io_st.cur_stack_size > 0) {
        uint32_t node_idx = io_st.node_stack[--io_st.cur_stack_size];

        bool is_leaf = i_dp.nodes[node_idx].is_leaf(); 
        if (tree_type == TreeType::NBVH) {
            is_leaf |= i_dp.nodes[node_idx].is_nbvh_leaf();
        }

        if (is_leaf) {
            const BVHNode &node = i_dp.nodes[node_idx];

            /* ==== intersect only node bbox ==== */
            if (mode == TraverseMode::ANOTHER_BBOX) {
                HitResult bbox_hit = ray_box_intersection(i_ray, node.bbox);
                bbox_hit.node_idx = node_idx;
                if (bbox_hit.hit) {
                    return bbox_hit;
                }
            }
            
            /* ==== intersect primitives in node ==== */
            else if (mode == TraverseMode::CLOSEST_PRIMITIVE) {
                HitResult bbox_hit = ray_box_intersection(i_ray, node.bbox);
                if (bbox_hit.t1 > closest_hit.t) {
                    continue;
                }

                HitResult node_hit = {false, FLT_MAX};
                for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                    uint32_t face_idx = i_dp.prim_idxs[prim_i];
                    const Face &face = i_dp.faces[face_idx];

                    HitResult prim_hit = ray_triangle_intersection(i_ray, face, i_dp.vertices);

                    if (prim_hit.hit && prim_hit.t < node_hit.t) {
                        node_hit = prim_hit;
                    }
                }

                node_hit.node_idx = node_idx;

                if (node_hit.hit && node_hit.t < closest_hit.t) {
                    closest_hit = node_hit;
                }
            }

            /* ==== idk who needs this but return the closest bbox ==== */
            else if (mode == TraverseMode::CLOSEST_BBOX) {
                HitResult bbox_hit = ray_box_intersection(i_ray, node.bbox);
                bbox_hit.node_idx = node_idx;
                if (bbox_hit.hit && bbox_hit.t1 < closest_hit.t1) {
                    closest_hit = bbox_hit;
                }
            }
        }

        /* ==== non-leaf case ==== */
        else {
            uint32_t left = i_dp.nodes[node_idx].left(node_idx);
            uint32_t right = i_dp.nodes[node_idx].right();

            HitResult left_hit = ray_box_intersection(i_ray, i_dp.nodes[left].bbox);
            HitResult right_hit = ray_box_intersection(i_ray, i_dp.nodes[right].bbox);

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
    }

    return closest_hit;
}