#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

#include "build.h"

BVHData CPUBuilder::build_bvh(int max_depth) {
    using LibVec3 = bvh::v2::Vec<float, 3>;
    using LibBBox = bvh::v2::BBox<float, 3>;
    using LibTri  = bvh::v2::Tri<float, 3>;
    using LibNode = bvh::v2::Node<float, 3>;
    using LibBVH  = bvh::v2::Bvh<LibNode>;

    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    int n_faces = faces.size();

    std::vector<LibBBox> bboxes(n_faces);
    std::vector<LibVec3> centers(n_faces);
    executor.for_each(0, n_faces, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const Face &face = faces[i];

            glm::vec3 centroid = face.get_centroid(vertices.data());
            BBox bounds = face.get_bounds(vertices.data());

            glm::vec3 min = bounds.min;
            glm::vec3 max = bounds.max;

            bboxes[i]  = {
                {min.x, min.y, min.z},
                {max.x, max.y, max.z}
            };
            centers[i] = {centroid.x, centroid.y, centroid.z};
        }
    });

    typename bvh::v2::DefaultBuilder<LibNode>::Config config;
    config.quality       = bvh::v2::DefaultBuilder<LibNode>::Quality::High;
    config.min_leaf_size = 1;
    config.max_leaf_size = 10;
    LibBVH lib_bvh = bvh::v2::DefaultBuilder<LibNode>::build(thread_pool, bboxes, centers, config);

    BVHData bvh_data;

    bvh_data.nodes.resize(lib_bvh.nodes.size());

    for (size_t node_id = 0; node_id < lib_bvh.nodes.size(); node_id++) {
        LibNode node                 = lib_bvh.nodes[node_id];
        LibBBox bbox                 = lib_bvh.nodes[node_id].get_bbox();
        bvh_data.nodes[node_id].bbox.min = glm::vec3(bbox.min[0], bbox.min[1], bbox.min[2]);
        bvh_data.nodes[node_id].bbox.max = glm::vec3(bbox.max[0], bbox.max[1], bbox.max[2]);

        bvh_data.nodes[node_id].n_prims         = node.index.prim_count();
        bvh_data.nodes[node_id].left_first_prim = node.index.first_id();
    }

    std::vector<Face> new_faces(n_faces);
    for (int i = 0; i < n_faces; i++) {
        int j = lib_bvh.prim_ids[i];
        new_faces[i] = faces[j];
    }
    bvh_data.faces = new_faces;
    bvh_data.vertices = vertices;
    bvh_data.n_nodes = lib_bvh.nodes.size();
    bvh_data.n_leaves = bvh_data.get_n_leaves();
    bvh_data.depth = bvh_data.get_depth();

    return bvh_data;
}

// thanks gpt-o1
void BVHData::save_as_obj(const std::string &filename, int max_depth) {
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
    std::function<void(uint32_t, int)> traverseAndWrite = [&](uint32_t node, int depth) {
        if (depth == 0) {
            writeCube(nodes[node].bbox.min, nodes[node].bbox.max);
            return;
        }
        if (nodes[node].is_leaf()) {
            writeCube(nodes[node].bbox.min, nodes[node].bbox.max);
        } else {
            traverseAndWrite(nodes[node].left(), depth - 1);
            traverseAndWrite(nodes[node].right(), depth - 1);
        }        
    };

    // Start traversal and writing
    traverseAndWrite(0, max_depth);

    outFile.close();
    std::cout << "BVH saved as .obj file: " << filename << std::endl;
}
