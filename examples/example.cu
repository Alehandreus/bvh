#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    int n_rays = img_size * img_size;
    int n_pixels = img_size * img_size;

    /* ==== Preparing ray data ==== */

    bvh.cudify();

    thrust::host_vector<glm::vec3> ray_origins(n_rays);
    thrust::host_vector<glm::vec3> ray_vectors(n_rays);
    thrust::host_vector<bool> masks(n_rays);
    thrust::host_vector<float> t(n_rays);

    cout << "Generating rays..." << endl;
    timer_start();
    #pragma omp parallel for
    for (int y = 0; y < img_size; y++) {
        for (int x = 0; x < img_size; x++) {
            float x_f = ((float)x / img_size - 0.5f) * 2;
            float y_f = ((float)y / img_size - 0.5f) * 2;

            glm::vec3 dir = cam_dir + x_dir * x_f + y_dir * y_f;
            
            ray_origins[y * img_size + x] = cam_pos;
            ray_vectors[y * img_size + x] = dir;
            masks[y * img_size + x] = true;
            t[y * img_size + x] = 0;
        }
    }
    cout << "Elapsed time: " << timer_stop() << " ms" << endl;

    thrust::device_vector<glm::vec3> ray_origins_d = ray_origins;
    thrust::device_vector<glm::vec3> ray_vectors_d = ray_vectors;
    thrust::device_vector<bool> masks_d = masks;
    thrust::device_vector<float> t_d = t;

    cout << endl;


    /* ==== Rendering image ==== */

    cout << "Rendering image..." << endl;
    timer_start();
    bvh.closest_primitive_cuda(
        ray_origins_d.data().get(),
        ray_vectors_d.data().get(),
        masks_d.data().get(),
        t_d.data().get(),
        n_rays
    );
    cout << "Elapsed time: " << timer_stop() << " ms" << endl;

    masks = masks_d;
    t = t_d;

    cout << endl;


    /* ==== Saving image ==== */

    std::vector<unsigned int> pixels(n_pixels);
    for (int i = 0; i < n_pixels; i++) {
        float val = 0;
        if (masks[i]) {
            val = std::sin(t[i] * glm::length(cam_dir) * 2) * 0.3f + 0.5f;
        }
        pixels[i] = (255 << 24) | ((int)(val * 255) << 16) | ((int)(val * 255) << 8) | (int)(val * 255);
    }
    cout << "Saving image..." << endl;
    SaveToBMP(pixels.data(), img_size, img_size, "output.bmp");

    return 0;
}