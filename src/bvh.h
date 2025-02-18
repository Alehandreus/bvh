#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/glm.hpp>
#include <omp.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <tuple>

#include "utils.h"

#ifdef CUDA_ENABLED
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

using std::cin, std::cout, std::endl;

struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<Face> faces;

    Mesh() {}

    // https://learnopengl.com/Model-Loading/Model
    void load_scene(const char *scene_path) {
        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(scene_path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            cout << "Assimp error: " << importer.GetErrorString() << endl;
            exit(1);
        }

        vertices.clear();
        faces.clear();

        for (int mesh_i = 0; mesh_i < scene->mNumMeshes; mesh_i++) {
            aiMesh *ai_mesh = scene->mMeshes[mesh_i];

            for (int vertex_i = 0; vertex_i < ai_mesh->mNumVertices; vertex_i++) {
                aiVector3D vertex = ai_mesh->mVertices[vertex_i];
                vertices.push_back(glm::vec3(vertex.x, vertex.y, vertex.z));
            }

            for (int face_i = 0; face_i < ai_mesh->mNumFaces; face_i++) {
                aiFace face = ai_mesh->mFaces[face_i];
                for (int index_i = 0; index_i < face.mNumIndices; index_i++) {
                    faces.push_back({face.mIndices[0], face.mIndices[1], face.mIndices[2]});
                }
            }
        }
    }

    void split_faces(float frac) {
        std::vector<float> extents;
        for (int i = 0; i < faces.size(); i++) {
            Face &face = faces[i];
            float extent = face.extent(vertices.data());
            extents.push_back(extent);
        }

        std::sort(extents.begin(), extents.end());

        float threshold = extents[extents.size() * frac];

        for (int face_i = 0; face_i < faces.size();) {
            Face face = faces[face_i];
            float extent = face.extent(vertices.data());

            if (extent > threshold) {
                glm::vec3 mid1 = (vertices[face[0]] + vertices[face[1]]) / 2.0f;
                glm::vec3 mid2 = (vertices[face[1]] + vertices[face[2]]) / 2.0f;
                glm::vec3 mid3 = (vertices[face[2]] + vertices[face[0]]) / 2.0f;

                Face new_face1 = {face[0], size(vertices), size(vertices) + 2};
                Face new_face2 = {size(vertices), face[1], size(vertices) + 1};
                Face new_face3 = {size(vertices) + 2, size(vertices) + 1, face[2]};
                Face new_face4 = {size(vertices), size(vertices) + 1, size(vertices) + 2};

                faces[face_i] = new_face1;
                faces.push_back(new_face2);
                faces.push_back(new_face3);
                faces.push_back(new_face4);

                vertices.push_back(mid1);
                vertices.push_back(mid2);
                vertices.push_back(mid3);
            } else {
                face_i++;
            }
        }
    }

    std::tuple<glm::vec3, glm::vec3> bounds() {
        glm::vec3 min(FLT_MAX);
        glm::vec3 max(-FLT_MAX);

        for (const glm::vec3 &vertex : vertices) {
            min = glm::min(min, vertex);
            max = glm::max(max, vertex);
        }

        return {min, max};
    }
};

struct BVHNode {
    BBox bbox;
    uint32_t left_first_prim;
    uint32_t n_prims;

    CUDA_HOST_DEVICE inline bool is_leaf() const {
        return n_prims > 0;
    }

    CUDA_HOST_DEVICE inline uint32_t left() const {
        return left_first_prim;
    }

    CUDA_HOST_DEVICE inline uint32_t right() const {
        return left_first_prim + 1;
    }

    CUDA_HOST_DEVICE bool inside(glm::vec3 point) const {
        return bbox.inside(point);
    }

    void update_bounds(const Face *faces, const glm::vec3 *vertices, const uint32_t *prim_idxs) {
        bbox.min = glm::vec3(FLT_MAX);
        bbox.max = glm::vec3(-FLT_MAX);

        for (int prim_i = left_first_prim; prim_i < left_first_prim + n_prims; prim_i++) {
            const Face &face = faces[prim_idxs[prim_i]];

            for (int j = 0; j < 3; j++) {
                const glm::vec3 &vertex = vertices[face[j]];
                bbox.update(vertex);
            }
        }
    }
};

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

enum TraverseMode {
    CLOSEST_PRIMITIVE,
    CLOSEST_BBOX,
    ANOTHER_BBOX    
};

CUDA_HOST_DEVICE
HitResult bvh_traverse(
    const Ray &ray,
    const BVHDataPointers &dp,
    StackInfo &st,
    TraverseMode mode
);

#ifdef CUDA_ENABLED
CUDA_GLOBAL
void closest_primitive_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors,
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_reserve,
    bool *masks,
    float *t,
    int n_rays
);
#endif

struct BVH {
    Mesh mesh;
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> prim_idxs;

    int depth;
    int n_nodes;
    int n_leaves;

    std::vector<uint32_t> stack;
    std::vector<int> stack_sizes;

    #ifdef CUDA_ENABLED
    thrust::device_vector<glm::vec3> vertices_cuda;
    thrust::device_vector<Face> faces_cuda;
    thrust::device_vector<BVHNode> nodes_cuda;
    thrust::device_vector<uint32_t> prim_idxs_cuda;
    thrust::device_vector<uint32_t> stack_cuda;
    thrust::device_vector<int> stack_sizes_cuda;
    #endif

    void load_scene(const char *path) {
        mesh.load_scene(path);
    }

    void split_faces(float frac) {
        mesh.split_faces(frac);
    }

    void build_bvh(int max_depth);
    void split_node(uint32_t node, int depth, int max_depth);
    
    #ifdef CUDA_ENABLED
    void cudify() {
        vertices_cuda = mesh.vertices;
        faces_cuda = mesh.faces;
        nodes_cuda = nodes;
        prim_idxs_cuda = prim_idxs;
        stack_cuda = stack;
        stack_sizes_cuda = stack_sizes;
    }
    #endif

    // save leaves as boxes in .obj file
    void save_as_obj(const std::string &filename);

    // traverse single ray, use local stack to be thread-safe
    HitResult closest_primitive(const Ray &ray) const {
        std::vector<uint32_t> smol_stack(depth * 2, 0);
        int smol_stack_size = 1;
        StackInfo stack_info = {smol_stack_size, smol_stack.data()};

        return bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_PRIMITIVE);
    }

    int memory_bytes() {
        return nodes.size() * sizeof(nodes[0]);
    }

    BVHDataPointers data_pointers() const {
        return {mesh.vertices.data(), mesh.faces.data(), nodes.data(), prim_idxs.data()};
    }

    #ifdef CUDA_ENABLED
    BVHDataPointers data_pointers_cuda() const {
        return {vertices_cuda.data().get(), faces_cuda.data().get(), nodes_cuda.data().get(), prim_idxs_cuda.data().get()};
    }
    #endif

    int stack_reserve() const {
        return depth * 2;
    }

    void reset_stack_batch(int n_rays) {
        stack.resize(n_rays * stack_reserve());
        std::fill(stack.begin(), stack.end(), 0);
        stack_sizes.resize(n_rays, 1);
        std::fill(stack_sizes.begin(), stack_sizes.end(), 1);
    }

    #ifdef CUDA_ENABLED
    void reset_stack_cuda(int n_rays) {
        stack_cuda.resize(n_rays * stack_reserve());
        thrust::fill(stack_cuda.begin(), stack_cuda.end(), 0);
        stack_sizes_cuda.resize(n_rays, 1);
        thrust::fill(stack_sizes_cuda.begin(), stack_sizes_cuda.end(), 1);
    }
    #endif

    // this and others are intended for python use, so structures like Ray and HitResult are not exposed
    void closest_primitive_batch(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        float *t,
        int n_rays
    ) {
        reset_stack_batch(n_rays);

        #pragma omp parallel for
        for (int i = 0; i < n_rays; i++) {
            Ray ray = {ray_origins[i], ray_vectors[i]};
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * stack_reserve()};

            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_PRIMITIVE);
            masks[i] = hit.hit;
            t[i] = hit.t;
        }
    }

    #ifdef CUDA_ENABLED
    void closest_primitive_cuda(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        float *t,
        int n_rays
    ) {
        reset_stack_cuda(n_rays);

        closest_primitive_entry
        <<< (n_rays + 31) / 32, 32 >>>
        (
            ray_origins,
            ray_vectors,
            data_pointers_cuda(),
            stack_cuda.data().get(),
            stack_sizes_cuda.data().get(),
            stack_reserve(),
            masks,
            t,
            n_rays
        );
    }
    #endif

    void closest_bbox_batch(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        uint32_t *node_idxs,
        float *t1,
        float *t2,
        int n_rays
    ) {
        reset_stack_batch(n_rays);

        #pragma omp parallel for
        for (int i = 0; i < n_rays; i++) {
            Ray ray = {ray_origins[i], ray_vectors[i]};
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * stack_reserve()};

            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_BBOX);
            masks[i] = hit.hit;
            node_idxs[i] = hit.node_idx;
            t1[i] = hit.t1;
            t2[i] = hit.t2;
        }
    }

    bool another_bbox_batch(
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
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * stack_reserve()};

            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::ANOTHER_BBOX);
            masks[i] = hit.hit;
            node_idxs[i] = hit.node_idx;
            t1[i] = hit.t1;
            t2[i] = hit.t2;

            alive = alive || hit.hit;
        }

        return alive;
    }

    // experiment for Transformer Model at github.com/Alehandreus/neural-intersection
    void segments_batch(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors, // ray_origins and ray_origins + ray_vectors are segment edges
        bool *segments,
        int n_rays,
        int n_segments
    ) {
        reset_stack_batch(n_rays);
        std::fill(segments, segments + n_rays * n_segments, false);
        float eps = 1e-6;

        #pragma omp parallel for
        for (int i = 0; i < n_rays; i++) {
            bool *cur_segments = segments + i * n_segments;

            Ray ray = {ray_origins[i], ray_vectors[i]};
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * stack_reserve()};

            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::ANOTHER_BBOX);
            while (hit.hit) {                
                float t1 = hit.t1;
                float t2 = hit.t2;

                t1 = std::max(t1, -eps);
                t2 = std::max(t2, -eps);

                uint32_t segment1 = (uint32_t) ((t1 - eps) * n_segments);
                uint32_t segment2 = (uint32_t) ((t2 + eps) * n_segments) + 1;

                segment1 = std::clamp(segment1, 0u, (uint32_t) n_segments - 1);
                segment2 = std::clamp(segment2, 1u, (uint32_t) n_segments);

                std::fill(cur_segments + segment1, cur_segments + segment2, true);

                hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::ANOTHER_BBOX);
            };
        }
    }
};
