#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cpu_traverse.h"

// #ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include "gpu_traverse.cuh"
#include "mesh_sampler.cuh"
// #endif

namespace nb = nanobind;

using h_float3 = nb::ndarray<float, nb::shape<3>, nb::numpy>;

using h_float2_batch = nb::ndarray<float, nb::shape<-1, 2>, nb::device::cpu, nb::c_contig>;
using h_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint3_batch = nb::ndarray<uint32_t, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uintN_batch = nb::ndarray<uint32_t, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>;
using h_int_batch = nb::ndarray<int, nb::shape<-1>, nb::device::cpu, nb::c_contig>;

using d_float2_batch = nb::ndarray<float, nb::shape<-1, 2>, nb::device::cuda, nb::c_contig>;
using d_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>;
using d_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_boolN_batch = nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cuda, nb::c_contig>;
using d_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_floatN_batch = nb::ndarray<float, nb::shape<-1, -1>, nb::device::cuda, nb::c_contig>;
using d_uint_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_uintN_batch = nb::ndarray<uint32_t, nb::shape<-1, -1>, nb::device::cuda, nb::c_contig>;
using d_int_batch = nb::ndarray<int, nb::shape<-1>, nb::device::cuda, nb::c_contig>;

NB_MODULE(mesh_utils_impl, m) {
    nb::class_<Mesh>(m, "Mesh")
        .def_static("from_file", [](const char *scene_path, bool swap_yz) {
            return Mesh(scene_path, swap_yz);
        })
        .def_static("from_data", [](const h_float3_batch &vertices, const h_uint3_batch &faces) {
            std::vector<glm::vec3> verts_vec(vertices.shape(0));
            std::memcpy(verts_vec.data(), vertices.data(), sizeof(glm::vec3) * vertices.shape(0));

            std::vector<Face> faces_vec(faces.shape(0));
            std::memcpy(faces_vec.data(), faces.data(), sizeof(Face) * faces.shape(0));

            return Mesh(std::move(verts_vec), std::move(faces_vec));
        })
        .def_static("from_data_with_uvs", [](const h_float3_batch &vertices, const h_uint3_batch &faces, const h_float2_batch &uvs) {
            std::vector<glm::vec3> verts_vec(vertices.shape(0));
            std::memcpy(verts_vec.data(), vertices.data(), sizeof(glm::vec3) * vertices.shape(0));

            std::vector<Face> faces_vec(faces.shape(0));
            std::memcpy(faces_vec.data(), faces.data(), sizeof(Face) * faces.shape(0));

            std::vector<glm::vec2> uvs_vec(uvs.shape(0));
            std::memcpy(uvs_vec.data(), uvs.data(), sizeof(glm::vec2) * uvs.shape(0));

            return Mesh(std::move(verts_vec), std::move(faces_vec), std::move(uvs_vec));
        })
        .def("get_num_vertices", [](Mesh& self) {
            return self.vertices.size();
        })
        .def("get_num_faces", [](Mesh& self) {
            return self.faces.size();
        })
        .def("get_vertices", [](Mesh& self) {
            return nb::ndarray<float, nb::numpy>(
                (float *) self.vertices.data(),
                {self.vertices.size(), 3}
            );
        })
        .def("get_faces", [](Mesh& self) {
            return nb::ndarray<uint32_t, nb::numpy>(
                (uint32_t *) self.faces.data(),
                {self.faces.size(), 3}
            );
        })
        .def("has_uvs", [](Mesh& self) {
            return !self.uvs.empty();
        })
        .def("get_uvs", [](Mesh& self) {
            return nb::ndarray<float, nb::numpy>(
                (float *) self.uvs.data(),
                {self.uvs.size(), 2}
            );
        })
        .def("save_preview", [](Mesh& self, const char *filename, int width, int height, h_float3& c_h, float R) {
            glm::vec3 c(c_h.data()[0], c_h.data()[1], c_h.data()[2]);
            return self.save_preview(filename, width, height, c, R);
        })
        .def("get_c", [](Mesh& self) {
            glm::vec3 c = self.get_c();
            return h_float3(&c).cast();
        })
        .def("get_R", &Mesh::get_R)
        .def("save_to_obj", &Mesh::save_to_obj)
        .def("split_faces", &Mesh::split_faces)
        .def("get_bounds", [](Mesh& self) {
            auto [min, max] = self.bounds();
            return nb::make_tuple(
                h_float3(&min).cast(),
                h_float3(&max).cast()
            );
        })
    ;

    nb::class_<BVHData>(m, "BVHData")
        .def_ro("depth", &BVHData::depth)
        .def_ro("n_nodes", &BVHData::n_nodes)
        .def_ro("n_leaves", &BVHData::n_leaves)
        .def("save_to_obj", [](BVHData& self, const char *filename, int max_depth) {
            self.save_to_obj(filename, max_depth);
        })
        .def("nodes_memory_bytes", &BVHData::nodes_memory_bytes)
        .def("nodes_data", [](BVHData& self) {
            glm::vec3 *min = new glm::vec3[self.nodes.size()];
            glm::vec3 *max = new glm::vec3[self.nodes.size()];

            for (int i = 0; i < self.nodes.size(); i++) {
                min[i] = self.nodes[i].bbox.min;
                max[i] = self.nodes[i].bbox.max;
            }

            auto min_arr = nb::ndarray<float, nb::numpy>(min, {self.nodes.size(), 3});
            auto max_arr = nb::ndarray<float, nb::numpy>(max, {self.nodes.size(), 3});

            return nb::make_tuple(min_arr, max_arr);
        })
        .def("get_faces", [](BVHData& self) {
            return nb::ndarray<uint32_t, nb::numpy>(
                (uint32_t *) self.faces.data(),
                {self.faces.size(), 3}
            );
        })
        .def("get_vertices", [](BVHData& self) {
            return nb::ndarray<float, nb::numpy>(
                (float *) self.vertices.data(),
                {self.vertices.size(), 3}
            );
        })
        .def("has_uvs", [](BVHData& self) {
            return !self.uvs.empty();
        })
        .def("get_uvs", [](BVHData& self) {
            return nb::ndarray<float, nb::numpy>(
                (float *) self.uvs.data(),
                {self.uvs.size(), 2}
            );
        })
    ;

    nb::class_<CPUBuilder>(m, "CPUBuilder")
        .def(nb::init<const Mesh&>())
        .def("build_bvh", &CPUBuilder::build_bvh, "Use mesh provided in constructor to build BVH with given depth. Returns BVHData.")
    ;

    nb::class_<CPUTraverser>(m, "CPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("ray_query", [](
            CPUTraverser& self,
            h_float3_batch& i_ray_origs,
            h_float3_batch& i_ray_vecs,
            h_bool_batch& o_mask,
            h_float_batch& o_t,
            h_uint_batch& o_prim_idx,
            h_float3_batch& o_normals,
            h_float2_batch& o_uvs,
            bool allow_negative
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.ray_query(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t.data(),
                o_prim_idx.data(),
                (glm::vec3 *) o_normals.data(),
                (glm::vec2 *) o_uvs.data(),
                n_rays,
                allow_negative
            );
        }, nb::arg("i_ray_origs"), nb::arg("i_ray_vecs"), nb::arg("o_mask"), nb::arg("o_t"),
           nb::arg("o_prim_idx"), nb::arg("o_normals"), nb::arg("o_uvs"), nb::arg("allow_negative") = false)
        .def("point_query", [](
            CPUTraverser& self,
            h_float3_batch& i_points,
            h_float_batch& o_t,
            h_float3_batch& o_closests
        ) {
            uint32_t n_points = i_points.shape(0);

            self.point_query(
                (glm::vec3 *) i_points.data(),
                o_t.data(),
                (glm::vec3 *) o_closests.data(),
                n_points
            );
        })
    ;

    // #ifdef CUDA_ENABLED
    nb::enum_<MeshSamplerMode>(m, "MeshSamplerMode")
        .value("SURFACE_UNIFORM", MeshSamplerMode::SURFACE_UNIFORM)
        .export_values()
    ;

    nb::class_<GPUMeshSampler>(m, "GPUMeshSampler")
        .def(nb::init<const Mesh&, MeshSamplerMode, int>(), nb::arg("mesh"), nb::arg("mode"), nb::arg("max_points"))
        .def("sample", [](
            GPUMeshSampler& self,
            d_float3_batch& o_points,
            d_float3_batch& o_coords,
            d_uint_batch& o_face_idxs,
            int n_points
        ) {
            if (n_points > self.max_points_) {
                throw std::runtime_error("n_points exceeds max_points set in constructor");
            }
            if (o_points.shape(0) < n_points || o_points.shape(1) != 3) {
                throw std::runtime_error("o_points has incorrect shape");
            }

            self.sample(
                (glm::vec3 *) o_points.data(),
                (glm::vec3 *) o_coords.data(),
                o_face_idxs.data(),
                n_points
            );
        })
    ;

    nb::class_<GPUTraverser>(m, "GPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("ray_query", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_bool_batch& o_mask,
            d_float_batch& o_t,
            d_uint_batch& o_prim_idx,
            d_float3_batch& o_normals,
            d_float2_batch& o_uvs,
            bool allow_negative
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.ray_query(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t.data(),
                o_prim_idx.data(),
                (glm::vec3 *) o_normals.data(),
                (glm::vec2 *) o_uvs.data(),
                n_rays,
                allow_negative
            );
        }, nb::arg("i_ray_origs"), nb::arg("i_ray_vecs"), nb::arg("o_mask"), nb::arg("o_t"),
           nb::arg("o_prim_idx"), nb::arg("o_normals"), nb::arg("o_uvs"), nb::arg("allow_negative") = false)
        .def("ray_query_all", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_boolN_batch& o_mask,
            d_floatN_batch& o_dist,
            d_uintN_batch& o_prim_idx,
            d_uint_batch& o_n_hits,
            int max_hits_per_ray,
            bool allow_negative
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.ray_query_all(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_dist.data(),
                o_prim_idx.data(),
                o_n_hits.data(),
                max_hits_per_ray,
                n_rays,
                allow_negative
            );
        }, nb::arg("i_ray_origs"), nb::arg("i_ray_vecs"), nb::arg("o_mask"), nb::arg("o_dist"),
           nb::arg("o_prim_idx"), nb::arg("o_n_hits"), nb::arg("max_hits_per_ray"), nb::arg("allow_negative") = false)        
        .def("point_query", [](
            GPUTraverser& self,
            d_float3_batch& i_points,
            d_float_batch& o_t,
            d_float3_batch& o_closests,
            d_float3_batch& o_barycentricses,
            d_uint_batch& o_face_idxs
        ) {
            uint32_t n_points = i_points.shape(0);

            self.point_query(
                (glm::vec3 *) i_points.data(),
                o_t.data(),
                (glm::vec3 *) o_closests.data(),
                (glm::vec3 *) o_barycentricses.data(),
                o_face_idxs.data(),
                n_points
            );
        })
    ;
    // #endif
}
