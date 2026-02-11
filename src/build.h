#pragma once

#include <fstream>
#include <string>
#include <functional>

#include "mesh.h"

struct BVHNode {
    BBox bbox;
    uint32_t left_first_prim;
    uint32_t n_prims : 31;
    uint32_t is_nbvh_leaf_ : 1;
    uint32_t father;

    CUDA_HOST_DEVICE inline bool is_leaf() const {
        return n_prims > 0;
    }

    CUDA_HOST_DEVICE inline bool is_nbvh_leaf() const {
        return is_leaf() || is_nbvh_leaf_;
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

    void update_bounds(const Face *faces, const glm::vec3 *vertices) {
        bbox.min = glm::vec3(FLT_MAX);
        bbox.max = glm::vec3(-FLT_MAX);

        for (int prim_i = left_first_prim; prim_i < left_first_prim + n_prims; prim_i++) {
            const Face &face = faces[prim_i];

            for (int j = 0; j < 3; j++) {
                const glm::vec3 &vertex = vertices[face[j]];
                bbox.update(vertex);
            }
        }
    }
};

struct BVHData {
    std::vector<Face> faces;  // Reordered faces for BVH traversal
    std::vector<BVHNode> nodes;

    int depth;
    int n_nodes;
    int n_leaves;

    // save leaves as boxes in .obj file
    void save_to_obj(const std::string &filename, int max_depth = -1);

    int nodes_memory_bytes() {
        return nodes.size() * sizeof(nodes[0]);
    }

    int get_depth(int cur_node = 0) const {
        if (nodes[cur_node].is_leaf()) {
            return 0;
        }
        return std::max(
            get_depth(nodes[cur_node].left()),
            get_depth(nodes[cur_node].right())
        ) + 1;
    }

    int get_n_leaves(int cur_node = 0) const {
        if (nodes[cur_node].is_leaf()) {
            return 1;
        }
        return get_n_leaves(nodes[cur_node].left()) + get_n_leaves(nodes[cur_node].right());
    }

private:
    BVHData() {}

    friend class CPUBuilder;
};

struct CPUBuilder {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<Face> faces;
    std::vector<Material> materials;
    std::vector<Texture> textures;

    CPUBuilder(const Mesh &mesh) : vertices(mesh.vertices), uvs(mesh.uvs), faces(mesh.faces),
                                     materials(mesh.materials), textures(mesh.textures) {}

    BVHData build_bvh(int max_leaf_size);
    void split_node(BVHData & bvh, uint32_t node, int cur_depth, int max_depth);
};
