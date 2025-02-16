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

#include "bvh.h"


void BVH::build_bvh(int max_depth) {
    int n_faces = mesh.faces.size();

    nodes.resize(n_faces * 2);

    for (int i = 0; i < mesh.faces.size(); i++) {
        prim_idxs.push_back(i);
    }

    for (int i = 0; i < n_faces; i++) {
        mesh.faces[i].calc_centroid(mesh.vertices.data());
    }
    
    BVHNode& root = nodes[0];
    root.left_first_prim = 0;
    root.n_prims = n_faces;
    root.update_bounds(mesh.faces.data(), mesh.vertices.data(), prim_idxs.data());

    n_nodes = 1;
    n_leaves = 0;
    depth = 1;

    grow_bvh(0, depth, max_depth);

    nodes.resize(n_nodes);
}

void BVH::grow_bvh(uint32_t node_idx, int cur_depth, int max_depth) {
    BVHNode &node = nodes[node_idx];

    depth = std::max(depth, cur_depth);
    if (node.n_prims <= 2 || cur_depth >= max_depth) {
        n_leaves++;
        return;
    }

    glm::vec3 extent = node.max - node.min;

    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    float splitPos = node.min[axis] + extent[axis] * 0.5f;

    // ingenious in-place partitioning
    int i = node.left_first_prim;
    int j = i + node.n_prims - 1;
    while (i <= j) {
        if (mesh.faces[prim_idxs[i]].centroid[axis] < splitPos) {
            i++;
        } else {
            std::swap(prim_idxs[i], prim_idxs[j--]);
        }            
    }

    int leftCount = i - node.left_first_prim;
    if (leftCount == 0 || leftCount == node.n_prims) {
        n_leaves++;
        return;
    }

    int left_idx = n_nodes++;
    int right_idx = n_nodes++;

    nodes[left_idx].left_first_prim = node.left_first_prim;
    nodes[left_idx].n_prims = leftCount;
    nodes[left_idx].update_bounds(mesh.faces.data(), mesh.vertices.data(), prim_idxs.data());

    nodes[right_idx].left_first_prim = i;
    nodes[right_idx].n_prims = node.n_prims - leftCount;
    nodes[right_idx].update_bounds(mesh.faces.data(), mesh.vertices.data(), prim_idxs.data());

    nodes[node_idx].left_first_prim = left_idx;
    nodes[node_idx].n_prims = 0;

    grow_bvh(left_idx, cur_depth + 1, max_depth);
    grow_bvh(right_idx, cur_depth + 1, max_depth);
}

std::tuple<bool, int, float, float> // mask, leaf index, t_enter, t_exit
BVH::intersect_leaves(
    const glm::vec3 &ray_origin,
    const glm::vec3 &ray_vector,
    int &stack_size,
    uint32_t *stack
) {
    if (stack_size == 1 && stack[0] == 0) {
        auto [mask, t1, t2] = ray_box_intersection(ray_origin, ray_vector, nodes[0].min, nodes[0].max);
        if (!mask) {
            return {false, -1, 0, 0};
        }
    }

    while (stack_size > 0) {
        uint32_t node_idx = stack[--stack_size];

        if (nodes[node_idx].is_leaf()) {
            auto [mask, t1, t2] = ray_box_intersection(ray_origin, ray_vector, nodes[node_idx].min, nodes[node_idx].max);
            return {mask, node_idx, t1, t2};
        }

        uint32_t left = nodes[node_idx].left();
        uint32_t right = nodes[node_idx].right();

        auto [mask_l, t1_l, t2_l] = ray_box_intersection(ray_origin, ray_vector, nodes[left].min, nodes[left].max);
        auto [mask_r, t1_r, t2_r] = ray_box_intersection(ray_origin, ray_vector, nodes[right].min, nodes[right].max);

        if (mask_l && mask_r && t1_l < t1_r) {
            std::swap(left, right);
            std::swap(t1_l, t1_r);
            std::swap(t2_l, t2_r);
        }

        if (mask_l) {
            stack[stack_size++] = left;
        }

        if (mask_r) {
            stack[stack_size++] = right;
        }
    }

    return {false, -1, 0, 0};
}

// thanks gpt-o1
void BVH::save_as_obj(const std::string &filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    int vertexOffset = 1; // Keeps track of vertex indices for face definitions

    // Lambda function to write a cube's vertices and faces
    auto writeCube = [&](const glm::vec3 &min, const glm::vec3 &max) {
        // Write vertices for the cube
        outFile << "v " << min.x << " " << min.y << " " << min.z << "\n"; // Bottom-left-front
        outFile << "v " << max.x << " " << min.y << " " << min.z << "\n"; // Bottom-right-front
        outFile << "v " << max.x << " " << max.y << " " << min.z << "\n"; // Top-right-front
        outFile << "v " << min.x << " " << max.y << " " << min.z << "\n"; // Top-left-front
        outFile << "v " << min.x << " " << min.y << " " << max.z << "\n"; // Bottom-left-back
        outFile << "v " << max.x << " " << min.y << " " << max.z << "\n"; // Bottom-right-back
        outFile << "v " << max.x << " " << max.y << " " << max.z << "\n"; // Top-right-back
        outFile << "v " << min.x << " " << max.y << " " << max.z << "\n"; // Top-left-back

        // Write faces for the cube (1-based indices)
        outFile << "f " << vertexOffset + 0 << " " << vertexOffset + 1 << " " << vertexOffset + 2 << " " << vertexOffset + 3 << "\n"; // Front
        outFile << "f " << vertexOffset + 4 << " " << vertexOffset + 5 << " " << vertexOffset + 6 << " " << vertexOffset + 7 << "\n"; // Back
        outFile << "f " << vertexOffset + 0 << " " << vertexOffset + 1 << " " << vertexOffset + 5 << " " << vertexOffset + 4 << "\n"; // Bottom
        outFile << "f " << vertexOffset + 3 << " " << vertexOffset + 2 << " " << vertexOffset + 6 << " " << vertexOffset + 7 << "\n"; // Top
        outFile << "f " << vertexOffset + 0 << " " << vertexOffset + 4 << " " << vertexOffset + 7 << " " << vertexOffset + 3 << "\n"; // Left
        outFile << "f " << vertexOffset + 1 << " " << vertexOffset + 5 << " " << vertexOffset + 6 << " " << vertexOffset + 2 << "\n"; // Right

        vertexOffset += 8; // Move to the next set of vertices
    };

    // Recursive function to traverse the BVH and write leaf nodes
    std::function<void(uint32_t)> traverseAndWrite = [&](uint32_t node) {
        if (nodes[node].is_leaf()) {
            writeCube(nodes[node].min, nodes[node].max);
        } else {
            traverseAndWrite(nodes[node].left());
            traverseAndWrite(nodes[node].right());
        }        
    };

    // Start traversal and writing
    traverseAndWrite(0);

    outFile.close();
    std::cout << "BVH saved as .obj file: " << filename << std::endl;
}


// experiment for Transformer Model at github.com/Alehandreus/neural-intersection
// BVH::intersect_leaves with modifications
void BVH::intersect_segments(const glm::vec3& start, const glm::vec3& end, int n_segments, bool* segments) {
    std::vector<uint32_t> stack(depth + 10, 0);
    int stack_size = 1;

    std::fill(segments, segments + n_segments, false);

    glm::vec3 o = start;
    glm::vec3 d = end - start;

    auto [mask, t1, t2] = ray_box_intersection(o, d, nodes[0].min, nodes[0].max);
    if (!mask) {
        return;
    }

    while (stack_size > 0) {
        uint32_t node_idx = stack[--stack_size];

        if (nodes[node_idx].is_leaf()) {
            auto [mask, t1, t2] = ray_box_intersection(o, d, nodes[node_idx].min, nodes[node_idx].max);

            // std::cout << "d1: " << t1 * glm::length(d) << "; d2: " << t2 * glm::length(d) << std::endl;

            float eps = 1e-3;

            t1 = std::max(t1, -eps);
            t2 = std::max(t2, -eps);

            uint32_t segment1 = (uint32_t) ((t1 - eps) * n_segments);
            uint32_t segment2 = (uint32_t) ((t2 + eps) * n_segments) + 1;

            // std::cout << "t1: " << t1 << "; t2: " << t2 << std::endl;
            // std::cout << "segment1: " << segment1 << "; segment2: " << segment2 << std::endl;

            // if (t1 < 0 || t2 < 0) {
            //     std::cout << "t1: " << t1 << "; t2: " << t2 << std::endl;
            // }

            segment1 = std::clamp(segment1, 0u, (uint32_t) n_segments - 1);
            segment2 = std::clamp(segment2, 1u, (uint32_t) n_segments);

            if (segment1 < segment2) {
                std::fill(segments + segment1, segments + segment2, true);
            } else {
                // std::cout << "t1: " << t1 << "; t2: " << t2 << std::endl;
                // std::cout << "segment1: " << segment1 << "; segment2: " << segment2 << std::endl;
                // what happened?
            }           

            continue;
        }

        uint32_t left = nodes[node_idx].left();
        uint32_t right = nodes[node_idx].right();

        auto [mask_l, t1_l, t2_l] = ray_box_intersection(o, d, nodes[left].min, nodes[left].max);
        auto [mask_r, t1_r, t2_r] = ray_box_intersection(o, d, nodes[right].min, nodes[right].max);

        if (mask_l) {
            stack[stack_size++] = left;
        }

        if (mask_r) {
            stack[stack_size++] = right;
        }
    }
}