#pragma once

#include <fstream>
#include <string>
#include <functional>

#include "mesh.h"

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

struct BVHData {
    std::vector<glm::vec3> vertices;
    std::vector<Face> faces;
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> prim_idxs;

    int depth;
    int n_nodes;
    int n_leaves;

    // save leaves as boxes in .obj file
    void save_as_obj(const std::string &filename);

    int nodes_memory_bytes() {
        return nodes.size() * sizeof(nodes[0]);
    }

private:
    BVHData() {}

    friend class CPUBuilder;
};

struct CPUBuilder {
    std::vector<glm::vec3> vertices;
    std::vector<Face> faces;

    CPUBuilder(const Mesh &mesh) : vertices(mesh.vertices), faces(mesh.faces) {}

    BVHData build_bvh(int max_depth);
    void split_node(BVHData & bvh, uint32_t node, int cur_depth, int max_depth);
};
