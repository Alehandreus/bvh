#include "cpu_traverse.h"

// at the moment cpu and gpu implementations are the same
// hence this file can be compiled by both nvcc and g++
CUDA_HOST_DEVICE HitResult bvh_traverse(
    const Ray &ray,
    const BVHDataPointers &dp,
    StackInfo &st,
    TraverseMode mode
) {
    HitResult closest_hit = {false, FLT_MAX, 0};

    while (st.stack_size > 0) {
        uint32_t node_idx = st.stack[--st.stack_size];

        if (dp.nodes[node_idx].is_leaf()) {
            const BVHNode &node = dp.nodes[node_idx];

            /* ==== intersect only node bbox ==== */
            if (mode == TraverseMode::ANOTHER_BBOX) {
                HitResult bbox_hit = ray_box_intersection(ray, node.bbox);
                if (bbox_hit.hit) {
                    return bbox_hit;
                }
            }
            
            /* ==== intersect primitives in node ==== */
            else if (mode == TraverseMode::CLOSEST_PRIMITIVE) {
                HitResult node_hit = {false, FLT_MAX};
                for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                    uint32_t face_idx = dp.prim_idxs[prim_i];
                    const Face &face = dp.faces[face_idx];

                    HitResult prim_hit = ray_triangle_intersection(ray, face, dp.vertices);

                    if (prim_hit.hit && prim_hit.t < node_hit.t) {
                        node_hit = prim_hit;
                    }
                }

                if (node_hit.hit && node_hit.t < closest_hit.t) {
                    closest_hit = node_hit;
                }
            }

            /* ==== idk who needs this but return the closest bbox ==== */
            else if (mode == TraverseMode::CLOSEST_BBOX) {
                HitResult bbox_hit = ray_box_intersection(ray, node.bbox);
                bbox_hit.node_idx = node_idx;
                if (bbox_hit.hit && bbox_hit.t1 < closest_hit.t1) {
                    closest_hit = bbox_hit;
                }
            }
        }

        /* ==== non-leaf case ==== */
        else {
            uint32_t left = dp.nodes[node_idx].left();
            uint32_t right = dp.nodes[node_idx].right();

            HitResult left_hit = ray_box_intersection(ray, dp.nodes[left].bbox);
            HitResult right_hit = ray_box_intersection(ray, dp.nodes[right].bbox);

            if (left_hit.hit) {
                st.stack[st.stack_size++] = left;
            }

            if (right_hit.hit) {
                st.stack[st.stack_size++] = right;
            }
        }
    }

    if (!closest_hit.hit) {
        closest_hit.t = 0;
    }

    return closest_hit;
}