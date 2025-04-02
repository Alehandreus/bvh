#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

#include "../src/cpu_traverse.h"

int main() {
    /* ==== Load and prepare BVH ==== */

    cout << "Loading scene..." << endl;
    Mesh mesh("suzanne.fbx");
    cout << "Number of vertices: " << mesh.vertices.size() << endl;
    cout << "Number of faces: " << mesh.faces.size() << endl;

    cout << endl;

    cout << "Splitting faces..." << endl;
    mesh.split_faces(0.9);
    cout << "Number of vertices: " << mesh.vertices.size() << endl;
    cout << "Number of faces: " << mesh.faces.size() << endl;

    cout << endl;
    
    CPUBuilder builder(mesh);
    cout << "Building BVH..." << endl;
    timer_start();
    BVHData bvh_data = builder.build_bvh(5);
    cout << "Elapsed time: " << timer_stop() << " ms" << endl;
    cout << "Depth: " << bvh_data.depth << endl;
    cout << "Number of nodes: " << bvh_data.n_nodes << endl;
    bvh_data.save_as_obj("bvh.obj");
    CPUTraverser bvh(bvh_data);

    cout << endl;

    /* ==== French bvh test ==== */

    using LibVec3 = bvh::v2::Vec<float, 3>;
    using LibBBox = bvh::v2::BBox<float, 3>;
    using LibTri  = bvh::v2::Tri<float, 3>;
    using LibNode = bvh::v2::Node<float, 3>;
    using LibBVH  = bvh::v2::Bvh<LibNode>;

    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    int n_faces = mesh.faces.size();

    std::vector<LibBBox> bboxes(n_faces);
    std::vector<LibVec3> centers(n_faces);
    executor.for_each(0, n_faces, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const Face &face = mesh.faces[i];

            glm::vec3 centroid = face.get_centroid(mesh.vertices.data());
            BBox bounds = face.get_bounds(mesh.vertices.data());

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
    LibBVH lib_bvh      = bvh::v2::DefaultBuilder<LibNode>::build(thread_pool, bboxes, centers, config);

    cout << "LibBVH built" << endl;

    for (auto &n : lib_bvh->nodes) {
        auto diagonal  = n.get_bbox().get_diagonal();
        auto bound_eps = BBOX_REGULARISATION;
        auto extent_eps =
            bvh::v2::Vec<float, 3>(bound_eps, bound_eps, bound_eps);  // n.get_bbox().get_diagonal() * bound_eps;
        for (size_t i = 0; i < 3; i++) {
            if (diagonal[i] <= BBOX_REGULARISATION) {
                n.bounds[2 * i] -= extent_eps[i] * 0.5f;      // min
                n.bounds[2 * i + 1] += extent_eps[i] * 0.5f;  // max
            }
        }
    }

        auto lib_root_bbox = lib_bvh_mesh->get_root().get_bbox();
        auto tmp_bbox      = Bbox3f{glm::vec3(lib_root_bbox.min[0], lib_root_bbox.min[1], lib_root_bbox.min[2]),
                                glm::vec3(lib_root_bbox.max[0], lib_root_bbox.max[1], lib_root_bbox.max[2])};

        m_cpu_mesh_bvhs[mesh_idx].build_from_libbvh(*lib_bvh_mesh);

        assert(m_cpu_mesh_bvhs[mesh_idx].root_bbox().min == tmp_bbox.min);
        assert(m_cpu_mesh_bvhs[mesh_idx].root_bbox().max == tmp_bbox.max);

        m_mesh_bvhs_nodes[mesh_idx].alloc_and_upload(m_cpu_mesh_bvhs[mesh_idx].m_nodes);
        m_mesh_bvhs_prim_indices[mesh_idx].alloc_and_upload(m_cpu_mesh_bvhs[mesh_idx].m_indices);

        cuda_mesh_bvhs[mesh_idx].vertices      = vertex_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].indices       = index_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].normals       = normal_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].texcoords     = texcoord_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].material_ids  = material_buffers[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].num_materials = material_buffers[mesh_idx].num_elements();
        cuda_mesh_bvhs[mesh_idx].material_map  = material_map_buffers[mesh_idx].d_pointer();

        cuda_mesh_bvhs[mesh_idx].nodes            = m_mesh_bvhs_nodes[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].num_nodes        = m_mesh_bvhs_nodes[mesh_idx].num_elements();
        cuda_mesh_bvhs[mesh_idx].prim_indices     = m_mesh_bvhs_prim_indices[mesh_idx].d_pointer();
        cuda_mesh_bvhs[mesh_idx].num_prim_indices = m_mesh_bvhs_prim_indices[mesh_idx].num_elements();

        if (m_neural_bvh_indices[0] == mesh_idx) {
            // neuralized mesh
            total_neuralized_vertex_buffer_size += mesh.vertices.size() * 4 * 3;
            total_neuralized_index_buffer_size += mesh.indices.size() * 4 * 3;
            total_neuralized_normal_buffer_size += mesh.normals.size() * 4 * 3;
            total_neuralized_texcoords_buffer_size += mesh.texcoords.size() * 4 * 2;
            total_neuralized_bvh_size += lib_bvh_mesh->nodes.size() * 32;
        } else {
            total_non_neuralized_vertex_buffer_size += mesh.vertices.size() * 4 * 3;
            total_non_neuralized_index_buffer_size += mesh.indices.size() * 4 * 3;
            total_non_neuralized_normal_buffer_size += mesh.normals.size() * 4 * 3;
            total_non_neuralized_texcoords_buffer_size += mesh.texcoords.size() * 4 * 2;
            total_non_neuralized_bvh_size += lib_bvh_mesh->nodes.size() * 32;
        }
    }
    // logger(LogLevel::Info, "Done!");

    // m_cuda_mesh_bvhs_buffer.alloc_and_upload(cuda_mesh_bvhs);

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
                float color = glm::dot(light_dir, hit.normal) * 0.5 + 0.5;
                img[y * img_size + x] = { color, color, color };
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
