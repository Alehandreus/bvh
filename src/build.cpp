#include "build.h"

BVHData CPUBuilder::build_bvh(int max_depth) {
    int n_faces = faces.size();

    BVHData bvh;
    bvh.vertices = vertices;
    bvh.faces = faces;
    bvh.nodes.resize(n_faces * 2);

    for (int i = 0; i < faces.size(); i++) {
        bvh.prim_idxs.push_back(i);
    }
    
    BVHNode& root = bvh.nodes[0];
    root.left_first_prim = 0;
    root.n_prims = n_faces;
    root.update_bounds(faces.data(), vertices.data(), bvh.prim_idxs.data());

    bvh.n_nodes = 1;
    bvh.n_leaves = 0;
    bvh.depth = 1;

    split_node(bvh, 0, 1, max_depth);

    bvh.nodes.resize(bvh.n_nodes);

    // rearrange primitives according to prim_idxs
    std::vector<Face> new_faces(n_faces);
    for (int i = 0; i < n_faces; i++) {
        new_faces[i] = faces[bvh.prim_idxs[i]];
    }
    bvh.faces = new_faces;

    return bvh;
}

void CPUBuilder::split_node(BVHData &bvh, uint32_t node_idx, int cur_depth, int max_depth) {
    BVHNode &node = bvh.nodes[node_idx];

    bvh.depth = std::max(bvh.depth, cur_depth);
    if (node.n_prims <= 2 || cur_depth >= max_depth) {
        bvh.n_leaves++;
        return;
    }

    glm::vec3 extent = node.bbox.diagonal();

    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    float splitPos = node.bbox.min[axis] + extent[axis] * 0.5f;

    // ingenious in-place partitioning
    int i = node.left_first_prim;
    int j = i + node.n_prims - 1;
    while (i <= j) {
        if (faces[bvh.prim_idxs[i]].get_centroid(bvh.vertices.data())[axis] < splitPos) {
            i++;
        } else {
            std::swap(bvh.prim_idxs[i], bvh.prim_idxs[j--]);
        }            
    }

    int leftCount = i - node.left_first_prim;
    if (leftCount == 0 || leftCount == node.n_prims) {
        bvh.n_leaves++;
        return;
    }

    int left_idx = bvh.n_nodes++;
    bvh.nodes[left_idx].left_first_prim = node.left_first_prim;
    bvh.nodes[left_idx].n_prims = leftCount;
    bvh.nodes[left_idx].update_bounds(faces.data(), vertices.data(), bvh.prim_idxs.data());
    bvh.nodes[left_idx].father = node_idx;

    split_node(bvh, left_idx, cur_depth + 1, max_depth);

    int right_idx = bvh.n_nodes++;
    bvh.nodes[right_idx].left_first_prim = i;
    bvh.nodes[right_idx].n_prims = node.n_prims - leftCount;
    bvh.nodes[right_idx].update_bounds(faces.data(), vertices.data(), bvh.prim_idxs.data());
    bvh.nodes[right_idx].father = node_idx;

    bvh.nodes[node_idx].left_first_prim = right_idx;
    bvh.nodes[node_idx].n_prims = 0;

    split_node(bvh, right_idx, cur_depth + 1, max_depth);
}

// thanks gpt-o1
void BVHData::save_as_obj(const std::string &filename) {
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
            writeCube(nodes[node].bbox.min, nodes[node].bbox.max);
        } else {
            traverseAndWrite(nodes[node].left(node));
            traverseAndWrite(nodes[node].right());
        }        
    };

    // Start traversal and writing
    traverseAndWrite(0);

    outFile.close();
    std::cout << "BVH saved as .obj file: " << filename << std::endl;
}
