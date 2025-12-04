#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cpu_traverse.h"

#include "gpu_traverse.cuh"

#define EPS 1e-6

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_states) {
        return;
    }
    curand_init(1234, i, 0, states + i);
}

CUDA_GLOBAL void ray_query_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_out,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];

    HitResult hit = ray_query(ray, i_dp, true);
    o_out.fill(i, hit);
}

CUDA_GLOBAL void point_query_entry(
    const glm::vec3 *i_points,
    const BVHDataPointers i_dp,
    SDFHitResults o_out,
    int n_points
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    glm::vec3 point = i_points[i];

    SDFHitResult hit = point_query(point, i_dp);
    o_out.fill(i, hit);
}

CUDA_DEVICE void ray_query_all_gpu(
    const Ray &i_ray,
    const BVHDataPointers &i_dp,
    HitResults o_hits,
    uint32_t &o_n_hits,
    int max_hits,
    bool allow_negative
) {
    uint32_t node_stack[TRAVERSE_STACK_SIZE];
    int cur_stack_size = 0;
    node_stack[cur_stack_size++] = 0; // root node

    int n_hits_registered = 0;

    while (cur_stack_size > 0 && n_hits_registered < max_hits) {
        uint32_t node_idx = node_stack[--cur_stack_size];
        const BVHNode &node = i_dp.nodes[node_idx];

        bool is_leaf = node.is_leaf();

        if (is_leaf) {
            HitResult node_hit = {false, FLT_MAX};
            for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                const Face &face = i_dp.faces[prim_i];

                HitResult prim_hit = ray_triangle_intersection(i_ray, face, i_dp.vertices, allow_negative);
                prim_hit.prim_idx = prim_i;

                if (prim_hit.hit) {
                    o_hits.fill(n_hits_registered++, prim_hit);
                    if (n_hits_registered >= max_hits) {
                        break;
                    }
                }
            }
        } else {
            uint32_t left = node.left();
            HitResult left_hit = ray_box_intersection(i_ray, i_dp.nodes[left].bbox, allow_negative);
            if (left_hit.hit) {
                node_stack[cur_stack_size++] = left;
            }

            uint32_t right = node.right();            
            HitResult right_hit = ray_box_intersection(i_ray, i_dp.nodes[right].bbox, allow_negative);
            if (right_hit.hit) {
                node_stack[cur_stack_size++] = right;
            }
        }
    }

    o_n_hits = n_hits_registered;

    // sort o_hits by t
    thrust::device_ptr<bool> mask_ptr(o_hits.masks);
    thrust::device_ptr<float> t_ptr(o_hits.t);

    thrust::sort_by_key(
        thrust::device,
        t_ptr,
        t_ptr + n_hits_registered,
        mask_ptr
    );

    // if (o_hits.prim_idxs) {
    //     thrust::device_ptr<uint32_t> prim_idx_ptr(o_hits.prim_idxs);
    //     thrust::sort_by_key(
    //         thrust::device,
    //         t_ptr,
    //         t_ptr + n_hits_registered,
    //         prim_idx_ptr
    //     );
    // }

    thrust::sort(
        thrust::device,
        t_ptr,
        t_ptr + n_hits_registered
    );
}

CUDA_GLOBAL void ray_query_all_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_hits,
    uint32_t *o_n_hits,
    int max_hits_per_ray,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];

    HitResults o_hits_moved = HitResults{
        o_hits.masks + i * max_hits_per_ray,
        o_hits.t + i * max_hits_per_ray,
        o_hits.prim_idxs + i * max_hits_per_ray,
        // o_hits.normals + i * max_hits_per_ray
    };

    ray_query_all_gpu(
        ray,
        i_dp,
        o_hits_moved,
        o_n_hits[i],
        max_hits_per_ray,
        false
    );
}
