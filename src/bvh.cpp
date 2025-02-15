// https://learnopengl.com/Model-Loading/Model

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


void BVH::build_bvh(int depth) {
    nodes.push_back(BVHNode());
    BVHNode *root = &nodes[0];

    root->min = glm::vec3(FLT_MAX);
    root->max = glm::vec3(-FLT_MAX);

    for (int i = 0; i < mesh.vertices.size(); i++) {
        glm::vec3 vertex = mesh.vertices[i];
        root->min = glm::min(root->min, vertex);
        root->max = glm::max(root->max, vertex);
    }

    root->faces = mesh.faces;

    grow_bvh(0, depth);
}


float leaf_cost(const BVHNode& node) {
    return node.faces.size() * TRIANGLE_INTERSECTION_COST;
}


float box_area(const glm::vec3& min, const glm::vec3& max) {
    glm::vec3 size = max - min;
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

float box_volume(const glm::vec3& min, const glm::vec3& max) {
    glm::vec3 size = max - min;
    return size.x * size.y * size.z;
}


float split_cost(const BVH& bvh, int node, int axis, const std::vector<Face>& faces_sorted, int split_i) {
    BVHNode left, right;

    for (int j = 0; j < 3; ++j) {
        left.min = glm::min(left.min, bvh.mesh.vertices[faces_sorted[0][j]]);
        left.max = glm::max(left.max, bvh.mesh.vertices[faces_sorted[split_i - 1][j]]);

        right.min = glm::min(right.min, bvh.mesh.vertices[faces_sorted[split_i][j]]);
        right.max = glm::max(right.max, bvh.mesh.vertices[faces_sorted[faces_sorted.size() - 1][j]]);
    }

    float parent_area = (bvh.nodes[node].max - bvh.nodes[node].min)[axis];
    float left_area = (left.max - left.min)[axis];
    float right_area = (right.max - right.min)[axis];

    // cout << "Split after " << split_i << ";" << endl;
    // cout << "left min: " << left.min.x << " " << left.min.y << " " << left.min.z << "; left max: " << left.max.x << " " << left.max.y << " " << left.max.z << endl;
    // cout << "right min: " << right.min.x << " " << right.min.y << " " << right.min.z << "; right max: " << right.max.x << " " << right.max.y << " " << right.max.z << endl;
    // cout << "cost: " << (left.max - left.min)[axis] << " " << (right.max - right.min)[axis] << endl;

    // return (left.max - left.min)[axis] + (right.max - right.min)[axis];
    // return box_area(left.min, left.max) + box_area(right.min, right.max);

    float left_cost = left_area / parent_area * split_i * TRIANGLE_INTERSECTION_COST;
    float right_cost = right_area / parent_area * (faces_sorted.size() - split_i) * TRIANGLE_INTERSECTION_COST;

    return TRAVERSAL_COST + left_cost + right_cost;
}


void BVH::grow_bvh(int node, int depth) {
    BVHNode *root = &nodes[node];

    if (depth <= 0 || root->faces.size() <= 1) {
        #ifdef DEBUG
        cout << "Child node min: " << root->min.x << " " << root->min.y << " " << root->min.z << endl;
        cout << "Child node max: " << root->max.x << " " << root->max.y << " " << root->max.z << endl;
        cout << endl;
        #endif

        return;
    }

    glm::vec3 size = root->max - root->min;
    int axis = 0;
    if (size.y > size.x && size.y > size.z) {
        axis = 1;
    }
    if (size.z > size.y && size.z > size.x) {
        axis = 2;
    }

    std::vector<Face> faces_sorted = root->faces;
    std::sort(faces_sorted.begin(), faces_sorted.end(), [&](Face a, Face b) {
        glm::vec3 a_min = glm::vec3(FLT_MAX);
        glm::vec3 a_max = glm::vec3(-FLT_MAX);
        glm::vec3 b_min = glm::vec3(FLT_MAX);
        glm::vec3 b_max = glm::vec3(-FLT_MAX);

        for (int i = 0; i < 3; i++) {
            glm::vec3 a_vertex = mesh.vertices[a[i]];
            a_min = glm::min(a_min, a_vertex);
            a_max = glm::max(a_max, a_vertex);

            glm::vec3 b_vertex = mesh.vertices[b[i]];
            b_min = glm::min(b_min, b_vertex);
            b_max = glm::max(b_max, b_vertex);
        }

        return a_min[axis] < b_min[axis];
    });

    float best_cost = FLT_MAX;
    int best_split_i = -1;
    float cur_cost = leaf_cost(*root);

    for (int i = 1; i < faces_sorted.size(); i++) {
        float cost = split_cost(*this, node, axis, faces_sorted, i);
        if (cost < best_cost) {
            best_cost = cost;
            best_split_i = i;
        }
    }

    int split_i = best_split_i;
    // int split_i = faces_sorted.size() / 2;

    #ifdef DEBUG
    cout << "Splitting at " << split_i << " out of " << faces_sorted.size() << endl;
    #endif

    BVHNode left;
    left.min = glm::vec3(FLT_MAX);
    left.max = glm::vec3(-FLT_MAX);

    BVHNode right;
    right.min = glm::vec3(FLT_MAX);
    right.max = glm::vec3(-FLT_MAX);

    for (int i = 0; i < split_i; i++) {
        Face face = faces_sorted[i];
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = mesh.vertices[face[j]];
            left.min = glm::min(left.min, vertex);
            left.max = glm::max(left.max, vertex);
        }
        left.faces.push_back(face);
    }

    for (int i = split_i; i < faces_sorted.size(); i++) {
        Face face = faces_sorted[i];
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = mesh.vertices[face[j]];
            right.min = glm::min(right.min, vertex);
            right.max = glm::max(right.max, vertex);
        }
        right.faces.push_back(face);
    }

    #ifdef DEBUG
    cout << "Current depth is " << depth << endl;
    cout << "Parent node min: " << root->min.x << " " << root->min.y << " " << root->min.z << endl;
    cout << "Parent node max: " << root->max.x << " " << root->max.y << " " << root->max.z << endl;
    float split_coord = mesh.vertices[faces_sorted[split_i].v1][axis];
    cout << "Splitting at " << split_coord << " on axis " << axis << endl;
    cout << "Number of faces left: " << left.faces.size() << endl;
    cout << "Number of faces right: " << right.faces.size() << endl;
    cout << endl;
    #endif

    if (left.faces.size() > 0) {
        nodes.push_back(left);
        nodes[node].left = nodes.size() - 1;
        grow_bvh(nodes.size() - 1, depth - 1);
    }

    if (right.faces.size() > 0) {
        nodes.push_back(right);
        nodes[node].right = nodes.size() - 1;
        grow_bvh(nodes.size() - 1, depth - 1);
    }
}


std::tuple<bool, int, float, float> // mask, leaf index, t_enter, t_exit
BVH::intersect_leaves(const glm::vec3& o, const glm::vec3& d, int& stack_size, uint32_t* stack) {
    if (stack_size == 1 && stack[0] == 0) {
        auto [mask, t1, t2] = ray_box_intersection(o, d, nodes[0].min, nodes[0].max);
        if (!mask) {
            return {false, -1, 0, 0};
        }
    }

    while (stack_size > 0) {
        uint32_t node_idx = stack[--stack_size];

        if (nodes[node_idx].is_leaf()) {
            // redundant computation, yes I know
            auto [mask, t1, t2] = ray_box_intersection(o, d, nodes[node_idx].min, nodes[node_idx].max);

            return {mask, node_idx, t1, t2};
        }

        uint32_t left = nodes[node_idx].left;
        uint32_t right = nodes[node_idx].right;

        auto [mask_l, t1_l, t2_l] = ray_box_intersection(o, d, nodes[left].min, nodes[left].max);
        auto [mask_r, t1_r, t2_r] = ray_box_intersection(o, d, nodes[right].min, nodes[right].max);

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


void BVH::save_as_obj(const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    int vertexOffset = 1; // Keeps track of vertex indices for face definitions

    // Lambda function to write a cube's vertices and faces
    auto writeCube = [&](const glm::vec3& min, const glm::vec3& max) {
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
    std::function<void(int)> traverseAndWrite = [&](int node) {
        if (node == -1) return;

        if (nodes[node].is_leaf()) {
            // Leaf node: write its bounding box as a cube
            writeCube(nodes[node].min, nodes[node].max);
        }

        // Recursively traverse children
        traverseAndWrite(nodes[node].left);
        traverseAndWrite(nodes[node].right);
    };

    // Start traversal and writing
    traverseAndWrite(0);

    outFile.close();
    std::cout << "BVH saved as .obj file: " << filename << std::endl;
}


std::tuple<bool, float, float> // mask, t_enter, t_exit
ray_box_intersection(const glm::vec3 &o, const glm::vec3 &d, const glm::vec3 &min, const glm::vec3 &max) {
    glm::vec3 t1 = (min - o) / d;
    glm::vec3 t2 = (max - o) / d;

    glm::vec3 tmin = glm::min(t1, t2);
    glm::vec3 tmax = glm::max(t1, t2);

    float t_enter = glm::max(tmin.x, glm::max(tmin.y, tmin.z));
    float t_exit = glm::min(tmax.x, glm::min(tmax.y, tmax.z));

    if (t_exit < 0 || t_enter > t_exit) {
        return {false, 0, 0};
    }

    return {true, t_enter, t_exit};
}

// experiment for Transformer Model at github.com/Alehandreus/neural-intersection
// BVH::intersect_leaves with modifications
int BVH::intersect_segments(const glm::vec3& start, const glm::vec3& end, int n_segments, bool* segments) {
    std::vector<uint32_t> stack(max_depth, 0);
    int stack_size = 1;
    uint32_t closest_segment = n_segments;

    std::fill(segments, segments + n_segments, false);

    glm::vec3 o = start;
    glm::vec3 d = end - start;

    auto [mask, t1, t2] = ray_box_intersection(o, d, nodes[0].min, nodes[0].max);
    if (!mask) {
        return closest_segment;
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
                closest_segment = std::min(closest_segment, segment1);
            } else {
                std::cout << "t1: " << t1 << "; t2: " << t2 << std::endl;
                std::cout << "segment1: " << segment1 << "; segment2: " << segment2 << std::endl;
                // what happened?
            }           

            continue;
        }

        uint32_t left = nodes[node_idx].left;
        uint32_t right = nodes[node_idx].right;

        auto [mask_l, t1_l, t2_l] = ray_box_intersection(o, d, nodes[left].min, nodes[left].max);
        auto [mask_r, t1_r, t2_r] = ray_box_intersection(o, d, nodes[right].min, nodes[right].max);

        if (mask_l) {
            stack[stack_size++] = left;
        }

        if (mask_r) {
            stack[stack_size++] = right;
        }
    }

    return closest_segment;
}