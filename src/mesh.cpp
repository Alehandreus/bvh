#include "mesh.h"
#include "build.h"

void Mesh::build_bvh_internal(int max_leaf_size) {
    // Create builder from current mesh state
    CPUBuilder builder(*this);

    // Build BVH (this creates a BVHData with reordered faces)
    bvh = std::make_unique<BVHData>(builder.build_bvh(max_leaf_size));
}
