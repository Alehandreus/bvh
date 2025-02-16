#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <functional>
#include <unordered_set>
#include <algorithm>

#include "utils.h"

using std::cin, std::cout, std::endl;

// https://learnopengl.com/Model-Loading/Model
struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<Face> faces;

    Mesh() {}

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
};

struct BVHNode {
    glm::vec3 min, max;
    uint32_t left_first_prim;
    uint32_t n_prims;

    BVHNode() {
        min = glm::vec3(FLT_MAX);
        max = glm::vec3(-FLT_MAX);
    }

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
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }

    void update_bounds(const Face *faces, const glm::vec3 *vertices, const uint32_t *prim_idxs) {
        min = glm::vec3(FLT_MAX);
        max = glm::vec3(-FLT_MAX);

        for (int prim_i = left_first_prim; prim_i < left_first_prim + n_prims; prim_i++) {
            const Face &face = faces[prim_idxs[prim_i]];

            for (int j = 0; j < 3; j++) {
                const glm::vec3 &vertex = vertices[face[j]];
                min = glm::min(min, vertex);
                max = glm::max(max, vertex);
            }
        }
    }
};

struct BVH {
    Mesh mesh;
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> prim_idxs;

    int depth;
    int n_nodes;
    int n_leaves;

    BVH() {}

    void load_scene(const char *path) {
        mesh.load_scene(path);
    }

    void build_bvh(int max_depth); // inits root and grows bvh
    void grow_bvh(uint32_t node, int depth, int max_depth); // recursive function to grow bvh

    // save leaves as boxes in .obj file
    void save_as_obj(const std::string &filename);
    
    std::tuple<bool, int, float, float> // mask, leaf index, t_enter, t_exit
    intersect_leaves(const glm::vec3 &ray_origin, const glm::vec3 &ray_direction, int& stack_size, uint32_t *stack); // bvh traversal, stack_size and stack are altered

    // experiment for Transformer Model at github.com/Alehandreus/neural-intersection
    void intersect_segments(const glm::vec3 &start, const glm::vec3 &end, int n_segments, bool *segments);
};
