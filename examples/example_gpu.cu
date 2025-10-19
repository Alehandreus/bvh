#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>

#include "../src/gpu_traverse.cuh"

int main() {
    /* ==== Load and prepare BVH ==== */

    cout << "Loading scene..." << endl;
    Mesh mesh("suzanne.fbx");
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
    BVHData bvh_data = builder.build_bvh(35);
    cout << "Elapsed time: " << timer_stop() << " ms" << endl;
    cout << "Number of nodes: " << bvh_data.n_nodes << endl;
    cout << "Number of leaves: " << bvh_data.n_leaves << endl;
    cout << "Depth: " << bvh_data.depth << endl;
    // bvh_data.save_to_obj("bvh.obj", 12);
    GPUTraverser bvh(bvh_data);

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
    glm::vec3 light_dir = glm::normalize(glm::vec3(1, -1, 1));

    int img_size = 800;
    int n_rays = img_size * img_size;
    int n_pixels = img_size * img_size;


    /* ==== Preparing ray data ==== */

    thrust::host_vector<glm::vec3> ray_origins(n_rays);
    thrust::host_vector<glm::vec3> ray_vectors(n_rays);
    thrust::host_vector<bool> masks(n_rays);
    thrust::host_vector<float> t(n_rays);
    thrust::host_vector<glm::vec3> normals(n_rays);

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

    thrust::device_vector<uint32_t> prim_idxs_d(n_rays);
    thrust::device_vector<glm::vec3> ray_origins_d = ray_origins;
    thrust::device_vector<glm::vec3> ray_vectors_d = ray_vectors;
    thrust::device_vector<bool> masks_d = masks;
    thrust::device_vector<float> t_d = t;
    thrust::device_vector<glm::vec3> normals_d(n_rays);

    cout << endl;


    /* ==== Rendering image ==== */

    cout << "Rendering image..." << endl;
    timer_start();
    bvh.ray_query(
        ray_origins_d.data().get(),
        ray_vectors_d.data().get(),
        masks_d.data().get(),
        t_d.data().get(),
        prim_idxs_d.data().get(),
        normals_d.data().get(),
        n_rays
    );
    cudaDeviceSynchronize();
    cout << "Elapsed time: " << timer_stop() << " ms" << endl;

    normals = normals_d;
    masks = masks_d;
    t = t_d;

    cout << endl;


    /* ==== Saving image ==== */

    cout << "Saving image..." << endl;
    std::vector<unsigned int> pixels(n_pixels);
    for (int i = 0; i < n_pixels; i++) {
        float color = 0;
        if (masks[i]) {
            color = glm::dot(light_dir, normals[i]) * 0.5f + 0.5f;
        }
        pixels[i] = (255 << 24) | ((int)(color * 255) << 16) | ((int)(color * 255) << 8) | (int)(color * 255);
    }
    stbi_write_png("output.png", img_size, img_size, 4, pixels.data(), img_size * 4);

    return 0;
}
