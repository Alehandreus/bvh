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
#include "cuda_compat.h"

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

                Face new_face1 = {face[0], vertices.size(), vertices.size() + 2};
                Face new_face2 = {vertices.size(), face[1], vertices.size() + 1};
                Face new_face3 = {vertices.size() + 2, vertices.size() + 1, face[2]};
                Face new_face4 = {vertices.size(), vertices.size() + 1, vertices.size() + 2};

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

    inline bool is_leaf() const {
        return n_prims > 0;
    }

    inline uint32_t left() const {
        return left_first_prim;
    }

    inline uint32_t right() const {
        return left_first_prim + 1;
    }

    bool inside(glm::vec3 point) const {
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
    ALL_BBOXES    
};

HitResult bvh_traverse(
    const Ray &ray,
    const BVHDataPointers &dp,
    StackInfo &st,
    TraverseMode mode
);

struct BVH {
    Mesh mesh;
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> prim_idxs;

    int depth;
    int n_nodes;
    int n_leaves;

    std::vector<uint32_t> stack;
    std::vector<int> stack_sizes;

    BVH() {}

    void load_scene(const char *path) {
        mesh.load_scene(path);
    }

    void split_faces(float frac) {
        mesh.split_faces(frac);
    }

    int memory_bytes() {
        return nodes.size() * sizeof(nodes[0]);
    }

    BVHDataPointers data_pointers() const {
        return {mesh.vertices.data(), mesh.faces.data(), nodes.data(), prim_idxs.data()};
    }

    void build_bvh(int max_depth);
    void split_node(uint32_t node, int depth, int max_depth);

    // save leaves as boxes in .obj file
    void save_as_obj(const std::string &filename);

    // traverse single ray, use local stack to be thread-safe
    HitResult traverse_primitives(const Ray &ray) const {
        std::vector<uint32_t> smol_stack(depth * 2, 0);
        int smol_stack_size = 1;
        StackInfo stack_info = {smol_stack_size, smol_stack.data()};
        HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_PRIMITIVE);
        return hit;
    }

    // this is intended for python use, so structures like Ray and HitResult are not exposed
    void traverse_primitives_batch(
        const glm::vec3 *ray_origins,
        const glm::vec3 *ray_vectors,
        bool *masks,
        float *t,
        int n_rays
    ) {
        int one_stack_size = depth * 2;
        stack.resize(n_rays * one_stack_size);
        stack_sizes.resize(n_rays, 1);

        #pragma omp parallel for
        for (int i = 0; i < n_rays; i++) {
            Ray ray = {ray_origins[i], ray_vectors[i]};
            StackInfo stack_info = {stack_sizes[i], stack.data() + i * one_stack_size};
            HitResult hit = bvh_traverse(ray, data_pointers(), stack_info, TraverseMode::CLOSEST_PRIMITIVE);
            masks[i] = hit.hit;
            t[i] = hit.t;
        }
    }
};
