#include "../src/bvh.h"

int main() {
    /* ==== BVH library methods ==== */
    BVH bvh;

    cout << "Loading scene..." << endl;
    bvh.load_scene("suzanne.fbx");
    cout << "Number of vertices: " << bvh.mesh.vertices.size() << endl;
    cout << "Number of faces: " << bvh.mesh.faces.size() << endl;

    cout << endl;

    cout << "Splitting faces..." << endl;
    bvh.split_faces(0.9);
    cout << "Number of vertices: " << bvh.mesh.vertices.size() << endl;
    cout << "Number of faces: " << bvh.mesh.faces.size() << endl;

    cout << endl;

    cout << "Building BVH..." << endl;
    bvh.build_bvh(15);
    cout << "Depth: " << bvh.depth << endl;
    cout << "Number of nodes: " << bvh.n_nodes << endl;
    cout << "Number of vertices: " << bvh.mesh.vertices.size() << endl;
    cout << "Number of faces: " << bvh.mesh.faces.size() << endl;
    bvh.save_as_obj("bvh.obj");

    cout << endl;


    /* ==== Setting up camera ==== */

    auto [min, max] = bvh.mesh.bounds();
    glm::vec3 center = (max + min) * 0.5f;
    float max_extent = std::fmax(max.x - min.x, std::fmax(max.y - min.y, max.z - min.z));
    glm::vec3 cam_pos = { 
        center.x + max_extent * 1.0,
        center.y - max_extent * 1.5,
        center.z + max_extent * 0.5
    };
    glm::vec3 cam_dir = (center - cam_pos) * 0.9f;
    glm::vec3 x_dir = glm::normalize(glm::cross(cam_dir, glm::vec3(0, 0, 1))) * (max_extent / 2);
    glm::vec3 y_dir = -glm::normalize(glm::cross(x_dir, cam_dir)) * (max_extent / 2);

    int img_size = 800;

    std::vector<glm::vec3> img(img_size * img_size, glm::vec3(0));


    /* ==== Rendering image ==== */

    cout << "Rendering image..." << endl;
    timer_start();
    #pragma omp parallel for
    for (int y = 0; y < img_size; y++) {
        for (int x = 0; x < img_size; x++) {
            float x_f = ((float)x / img_size - 0.5f) * 2;
            float y_f = ((float)y / img_size - 0.5f) * 2;

            glm::vec3 dir = cam_dir + x_dir * x_f + y_dir * y_f;
            HitResult hit = bvh.closest_primitive({cam_pos, dir});

            if (hit.hit) {
                float val = std::sin(hit.t * glm::length(cam_dir) * 2) * 0.3f + 0.5f;
                img[y * img_size + x] = { val, val, val };
            }
        }
    }
    cout << "Elapsed time: " << timer_stop() << " ms" << endl;

    cout << endl;


    /* ==== Saving image ==== */

    std::vector<unsigned int> pixels(img_size * img_size);
    for (int i = 0; i < img_size * img_size; i++) {
        pixels[i] = (255 << 24) | ((int)(img[i].z * 255) << 16) | ((int)(img[i].y * 255) << 8) | (int)(img[i].x * 255);
    }
    cout << "Saving image..." << endl;
    save_to_bmp(pixels.data(), img_size, img_size, "output.bmp");

    return 0;
}