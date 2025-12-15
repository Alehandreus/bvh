#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <curand_kernel.h>

#include "mesh.h"
#include "utils.h"
#include "gpu_traverse.cuh"
#include "mesh_sampler.cuh"

CUDA_GLOBAL void mesh_sample_surface_uniform_entry(
    const glm::vec3 *i_vertices,
    const Face *i_faces,
    const float *i_face_weights_prefix_sum,
    int n_faces,
    curandState *io_rand_states,
    glm::vec3 *o_out_points,
    glm::vec3 *o_out_barycentrics,
    uint32_t *o_out_face_idxs,    
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    curandState local_state = io_rand_states[idx];

    // Sample a face based on the face weights
    float r = curand_uniform(&local_state);
    int face_idx = thrust::upper_bound(thrust::seq, i_face_weights_prefix_sum, i_face_weights_prefix_sum + n_faces, r) - i_face_weights_prefix_sum;
    face_idx = min(face_idx, n_faces - 1);

    Face face = i_faces[face_idx];
    glm::vec3 v0 = i_vertices[face.v1];
    glm::vec3 v1 = i_vertices[face.v2];
    glm::vec3 v2 = i_vertices[face.v3];

    // Sample a point uniformly within the triangle
    float u = curand_uniform(&local_state);
    float v = curand_uniform(&local_state);

    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }

    // glm::vec3 sampled_point = (1 - u - v) * v0 + u * v1 + v * v2;
    glm::vec3 sampled_point = {
        (1 - u - v) * v0.x + u * v1.x + v * v2.x,
        (1 - u - v) * v0.y + u * v1.y + v * v2.y,
        (1 - u - v) * v0.z + u * v1.z + v * v2.z
    };

    o_out_points[idx] = sampled_point;
    io_rand_states[idx] = local_state;

    // compute barycentric coordinates
    if (o_out_barycentrics != nullptr) {
        o_out_barycentrics[idx] = glm::vec3(1 - u - v, u, v);
    }

    // output face index
    if (o_out_face_idxs != nullptr) {
        o_out_face_idxs[idx] = face_idx;
    }
}