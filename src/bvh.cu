#include "bvh.h"

void BVH::build_bvh(int max_depth) {
    int n_faces = mesh.faces.size();

    nodes.resize(n_faces * 2);

    for (int i = 0; i < mesh.faces.size(); i++) {
        prim_idxs.push_back(i);
    }
    
    BVHNode& root = nodes[0];
    root.left_first_prim = 0;
    root.n_prims = n_faces;
    root.update_bounds(mesh.faces.data(), mesh.vertices.data(), prim_idxs.data());

    n_nodes = 1;
    n_leaves = 0;
    depth = 1;

    split_node(0, depth, max_depth);

    nodes.resize(n_nodes);
}

void BVH::split_node(uint32_t node_idx, int cur_depth, int max_depth) {
    BVHNode &node = nodes[node_idx];

    depth = std::max(depth, cur_depth);
    if (node.n_prims <= 2 || cur_depth >= max_depth) {
        n_leaves++;
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

    split_node(left_idx, cur_depth + 1, max_depth);
    split_node(right_idx, cur_depth + 1, max_depth);
}

CUDA_HOST_DEVICE
HitResult bvh_traverse(
    const Ray &ray,
    const BVHDataPointers &dp,
    StackInfo &st,
    TraverseMode mode
) {
    if (st.stack_size == 1 && st.stack[0] == 0) {
        auto hit = ray_box_intersection(ray, dp.nodes[0].bbox);
        if (!hit.hit) {
            return hit;
        }
    }

    HitResult closest_hit = {false, FLT_MAX, 0};

    while (st.stack_size > 0) {
        uint32_t node_idx = st.stack[--st.stack_size];

        if (dp.nodes[node_idx].is_leaf()) {
            const BVHNode &node = dp.nodes[node_idx];

            /* ==== intersect only node bbox ==== */
            if (mode == TraverseMode::ANOTHER_BBOX) {
                HitResult bbox_hit = ray_box_intersection(ray, node.bbox);
                if (bbox_hit.hit) {
                    return bbox_hit;
                }
            }
            
            /* ==== intersect primitives in node ==== */
            else if (mode == TraverseMode::CLOSEST_PRIMITIVE) {
                HitResult node_hit = {false, FLT_MAX};
                for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                    uint32_t face_idx = dp.prim_idxs[prim_i];
                    const Face &face = dp.faces[face_idx];

                    HitResult prim_hit = ray_triangle_intersection(ray, face, dp.vertices);

                    if (prim_hit.hit && prim_hit.t < node_hit.t) {
                        node_hit = prim_hit;
                    }
                }

                if (node_hit.hit && node_hit.t < closest_hit.t) {
                    closest_hit = node_hit;
                }
            }

            /* ==== idk who needs this but return the closest bbox ==== */
            else if (mode == TraverseMode::CLOSEST_BBOX) {
                HitResult bbox_hit = ray_box_intersection(ray, node.bbox);
                bbox_hit.node_idx = node_idx;
                if (bbox_hit.hit && bbox_hit.t1 < closest_hit.t1) {
                    closest_hit = bbox_hit;
                }
            }
        }

        /* ==== non-leaf case ==== */
        else {
            uint32_t left = dp.nodes[node_idx].left();
            uint32_t right = dp.nodes[node_idx].right();

            HitResult left_hit = ray_box_intersection(ray, dp.nodes[left].bbox);
            HitResult right_hit = ray_box_intersection(ray, dp.nodes[right].bbox);

            if (left_hit.hit) {
                st.stack[st.stack_size++] = left;
            }

            if (right_hit.hit) {
                st.stack[st.stack_size++] = right;
            }
        }
    }

    if (!closest_hit.hit) {
        closest_hit.t = 0;
    }

    return closest_hit;
}

#ifdef CUDA_ENABLED
CUDA_GLOBAL void closest_primitive_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors,
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_reserve,
    bool *masks,
    float *t,
    int n_rays
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    Ray ray = {ray_origins[i], ray_vectors[i]};
    StackInfo stack_info = {stack_sizes[i], stack + i * stack_reserve};

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::CLOSEST_PRIMITIVE);
    masks[i] = hit.hit;
    t[i] = hit.t;
}

CUDA_GLOBAL void segments_entry(
    const glm::vec3 *ray_origins,
    const glm::vec3 *ray_vectors, // ray_origins and ray_origins + ray_vectors are segment edges
    const BVHDataPointers dp,
    uint32_t *stack,
    int *stack_sizes,
    int stack_reserve,
    bool *segments,
    int n_rays,
    int n_segments
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }

    Ray ray = {ray_origins[i], ray_vectors[i]};
    StackInfo stack_info = {stack_sizes[i], stack + i * stack_reserve};
    bool *cur_segments = segments + i * n_segments;

    float eps = 1e-6;

    HitResult hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX);
    while (hit.hit) {                
        float t1 = hit.t1;
        float t2 = hit.t2;

        t1 = fmaxf(t1, -eps);
        t2 = fmaxf(t2, -eps);

        uint32_t segment1 = (uint32_t) ((t1 - eps) * n_segments);
        uint32_t segment2 = (uint32_t) ((t2 + eps) * n_segments) + 1;

        segment1 = fmaxf(segment1, 0u);
        segment1 = fminf(segment1, (uint32_t) n_segments);

        segment2 = fmaxf(segment2, 1u);
        segment2 = fminf(segment2, (uint32_t) n_segments);

        for (int j = segment1; j < segment2; j++) {
            cur_segments[j] = true;
        }

        hit = bvh_traverse(ray, dp, stack_info, TraverseMode::ANOTHER_BBOX);
    };    
}
#endif

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
            writeCube(nodes[node].bbox.min, nodes[node].bbox.max);
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
