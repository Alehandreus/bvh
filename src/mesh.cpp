#include "mesh.h"
// #include "build.h"

void Mesh::build_bvh_internal(int max_leaf_size) {
    bvh = std::make_unique<BVHData>(build_bvh(vertices, faces, max_leaf_size));

    // Replace mesh faces with BVH-reordered faces so face indices from
    // ray tracing match the faces returned by get_faces()
    this->faces = std::move(bvh->reordered_faces);
}
