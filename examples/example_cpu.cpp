#include "../src/cpu_traverse.h"

int main() {
    /* ==== Load and prepare BVH ==== */

    cout << "Loading scene..." << endl;
    Mesh mesh("models/lego.fbx");
    cout << "Number of vertices: " << mesh.vertices.size() << endl;
    cout << "Number of faces: " << mesh.faces.size() << endl;

    cout << endl;

    cout << "Splitting faces..." << endl;
    // mesh.split_faces(0.9);
    cout << "Number of vertices: " << mesh.vertices.size() << endl;
    cout << "Number of faces: " << mesh.faces.size() << endl;

    cout << endl;

    CPUBuilder builder(mesh);
    cout << "Building BVH..." << endl;
    timer_start();
    BVHData bvh_data = builder.build_bvh(5);
    cout << "Elapsed time: " << timer_stop() << " ms" << endl;
    cout << "Number of nodes: " << bvh_data.n_nodes << endl;
    cout << "Number of leaves: " << bvh_data.n_leaves << endl;
    cout << "Depth: " << bvh_data.depth << endl;
    bvh_data.save_as_obj("bvh.obj", 12);
    CPUTraverser bvh(bvh_data);

    cout << endl;

    /* ==== Setting up camera ==== */

    auto [min, max] = mesh.bounds();
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
    glm::vec3 light_dir = glm::normalize(glm::vec3(1, -1, 1));


    /* ==== Rendering image ==== */

    cout << "Rendering image..." << endl;
    timer_start();
    // #pragma omp parallel for
    for (int y = 0; y < img_size; y++) {
        for (int x = 0; x < img_size; x++) {
            float x_f = ((float)x / img_size - 0.5f) * 2;
            float y_f = ((float)y / img_size - 0.5f) * 2;

            glm::vec3 dir = cam_dir + x_dir * x_f + y_dir * y_f;
            HitResult hit = bvh.closest_primitive_single({cam_pos, dir});

            if (hit.hit) {
                float att = glm::dot(light_dir, hit.normal) * 0.5 + 0.5;
                glm::vec3 color = hit.color * att;
                img[y * img_size + x] = color;
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
