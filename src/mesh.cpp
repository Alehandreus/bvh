#include "mesh.h"
#include "build.h"

void Mesh::build_bvh_internal(int max_leaf_size) {
    // Create builder from current mesh state
    CPUBuilder builder(*this);

    // Build BVH (this creates a BVHData with reordered faces)
    bvh = std::make_unique<BVHData>(builder.build_bvh(max_leaf_size));

    // Replace mesh faces with BVH-reordered faces so face indices from
    // ray tracing match the faces returned by get_faces()
    this->faces = std::move(bvh->reordered_faces);
}
