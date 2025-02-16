#include "bvh.h"

int main() {
    BVH bvh;
    bvh.load_scene("lego.stl");
    cout << "Number of vertices: " << bvh.mesh.vertices.size() << endl;
    cout << "Number of faces: " << bvh.mesh.faces.size() << endl;

    cout << "Splitting faces..." << endl;
    bvh.split_faces(0.9);
    cout << "Number of vertices: " << bvh.mesh.vertices.size() << endl;
    cout << "Number of faces: " << bvh.mesh.faces.size() << endl;

    bvh.build_bvh(15);
    bvh.save_as_obj("bvh.obj");

    cout << "Depth: " << bvh.depth << endl;
    cout << "Number of nodes: " << bvh.n_nodes << endl;
    cout << "Number of vertices: " << bvh.mesh.vertices.size() << endl;
    cout << "Number of faces: " << bvh.mesh.faces.size() << endl;

    return 0;
}
