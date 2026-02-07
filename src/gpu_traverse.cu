#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cpu_traverse.h"

#include "gpu_traverse.cuh"

#define EPS 1e-6

// Debug flag - set by kernel for specific rays
CUDA_DEVICE bool g_debug_sample = false;

// GPU texture sampling implementation
CUDA_DEVICE glm::vec3 sampleTextureGPU(
    const TextureDeviceView& texture,
    float u, float v
) {
    // Handle empty texture
    if (!texture.pixels || texture.width == 0 || texture.height == 0) {
        return glm::vec3(1.0f);  // Default white (matches CPU behavior)
    }

    // Wrap UV coordinates to [0, 1)
    u = u - floorf(u);
    v = v - floorf(v);

    // Flip V axis (GLTF/OpenGL convention)
    v = 1.0f - v;

    // Calculate pixel coordinates
    float x = u * (texture.width - 1);
    float y = v * (texture.height - 1);

    // Get integer coordinates and interpolation factors
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, texture.width - 1);
    int y1 = min(y0 + 1, texture.height - 1);

    float tx = x - x0;
    float ty = y - y0;

    // Debug: print raw pixel values
    if (g_debug_sample) {
        size_t idx = (y0 * texture.width + x0) * 4;
        printf("  sampleTextureGPU: uv=(%.3f,%.3f) -> pixel(%d,%d), idx=%zu\n", u, v, x0, y0, idx);
        printf("  sampleTextureGPU: raw RGBA = [%d, %d, %d, %d]\n",
               texture.pixels[idx+0], texture.pixels[idx+1], texture.pixels[idx+2], texture.pixels[idx+3]);
    }

    // Fetch pixel helper
    auto fetchPixel = [&](int xi, int yi) -> glm::vec3 {
        size_t idx = (yi * texture.width + xi) * 4;
        return glm::vec3(
            texture.pixels[idx + 0] / 255.0f,  // R
            texture.pixels[idx + 1] / 255.0f,  // G
            texture.pixels[idx + 2] / 255.0f   // B
        );
    };

    // Fetch 4 corner pixels
    glm::vec3 c00 = fetchPixel(x0, y0);
    glm::vec3 c10 = fetchPixel(x1, y0);
    glm::vec3 c01 = fetchPixel(x0, y1);
    glm::vec3 c11 = fetchPixel(x1, y1);

    // Bilinear interpolation
    glm::vec3 c0 = glm::mix(c00, c10, tx);
    glm::vec3 c1 = glm::mix(c01, c11, tx);
    glm::vec3 result = glm::mix(c0, c1, ty);

    // Convert from sRGB to linear
    return srgbToLinear(result);
}

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
    const TextureDeviceView *texture_views,
    const MaterialDeviceView *materials,
    int n_textures,
    HitResults o_out,
    int n_rays,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];

    HitResult hit = ray_query(ray, i_dp, allow_negative, allow_backward, allow_forward);

    // Debug: Print texture info from first thread only
    if (i == 0) {
        printf("GPU Kernel: n_textures=%d, materials=%p, texture_views=%p, uvs=%p\n",
               n_textures, (void*)materials, (void*)texture_views, (void*)i_dp.uvs);
        if (texture_views && n_textures > 0) {
            printf("GPU Kernel: texture[0]: width=%d, height=%d, pixels=%p\n",
                   texture_views[0].width, texture_views[0].height,
                   (void*)texture_views[0].pixels);
        }
    }

    // Sample texture color on GPU
    if (hit.hit && materials && texture_views && i_dp.uvs) {
        const Face &face = i_dp.faces[hit.prim_idx];
        if (i == 320000) {  // Pick a ray in the middle that likely hits
            printf("GPU Kernel: Ray %d hit face.material_idx=%d, uv=(%.3f, %.3f)\n",
                   i, face.material_idx, hit.uv.x, hit.uv.y);
        }
        if (face.material_idx >= 0) {
            const MaterialDeviceView& mat = materials[face.material_idx];

            // Use texture color if available, otherwise use base color
            if (mat.texture_id >= 0 && mat.texture_id < n_textures) {
                g_debug_sample = (i == 320000);
                hit.color = sampleTextureGPU(
                    texture_views[mat.texture_id],
                    hit.uv.x,
                    hit.uv.y
                );
                g_debug_sample = false;
                if (i == 320000) {
                    printf("GPU Kernel: Ray %d sampled color=(%.3f, %.3f, %.3f)\n",
                           i, hit.color.r, hit.color.g, hit.color.b);
                }
            } else {
                hit.color = mat.base_color;
                if (i == 320000) {
                    printf("GPU Kernel: Ray %d using base_color (no texture)\n", i);
                }
            }
        }
    } else if (hit.hit && i == 320000) {
        printf("GPU Kernel: Ray %d hit but skipping color - materials=%p, texture_views=%p, uvs=%p\n",
               i, (void*)materials, (void*)texture_views, (void*)i_dp.uvs);
    }

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
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
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

                glm::vec3 normal = ray_triangle_norm(face, i_dp.vertices);
                float facing = vdot(normal, i_ray.vector);
                if (facing > 0.0f && !allow_backward) {
                    continue;
                }
                if (facing < 0.0f && !allow_forward) {
                    continue;
                }

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

    // Manual bubble/insertion sort - no temporary buffers needed
    // For small n_hits_registered (< 100), this is often faster than thrust
    for (int i = 1; i < n_hits_registered; i++) {
        float key_t = o_hits.t[i];
        bool key_mask = o_hits.masks[i];
        uint32_t key_prim = o_hits.prim_idxs ? o_hits.prim_idxs[i] : 0;
        
        int j = i - 1;
        while (j >= 0 && o_hits.t[j] > key_t) {
            o_hits.t[j + 1] = o_hits.t[j];
            o_hits.masks[j + 1] = o_hits.masks[j];
            if (o_hits.prim_idxs) {
                o_hits.prim_idxs[j + 1] = o_hits.prim_idxs[j];
            }
            j--;
        }
        o_hits.t[j + 1] = key_t;
        o_hits.masks[j + 1] = key_mask;
        if (o_hits.prim_idxs) {
            o_hits.prim_idxs[j + 1] = key_prim;
        }
    }
}

CUDA_GLOBAL void ray_query_all_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_hits,
    uint32_t *o_n_hits,
    int max_hits_per_ray,
    int n_rays,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
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
        allow_negative,
        allow_backward,
        allow_forward
    );
}
