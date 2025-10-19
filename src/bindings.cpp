#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cpu_traverse.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include "gpu_traverse.cuh"
#include "mesh_sampler.cuh"
#endif

namespace nb = nanobind;

using h_float3 = nb::ndarray<float, nb::shape<3>, nb::numpy>;

using h_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint3_batch = nb::ndarray<uint32_t, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uintN_batch = nb::ndarray<uint32_t, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>;
using h_int_batch = nb::ndarray<int, nb::shape<-1>, nb::device::cpu, nb::c_contig>;

using d_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>;
using d_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_uint_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_uintN_batch = nb::ndarray<uint32_t, nb::shape<-1, -1>, nb::device::cuda, nb::c_contig>;
using d_int_batch = nb::ndarray<int, nb::shape<-1>, nb::device::cuda, nb::c_contig>;

NB_MODULE(bvh_impl, m) {
    nb::class_<Mesh>(m, "Mesh")
        .def_static("from_file", [](const char *scene_path) {
            return Mesh(scene_path);
        })
        .def_static("from_data", [](const h_float3_batch &vertices, const h_uint3_batch &faces) {
            std::vector<glm::vec3> verts_vec(vertices.shape(0));
            std::memcpy(verts_vec.data(), vertices.data(), sizeof(glm::vec3) * vertices.shape(0));

            std::vector<Face> faces_vec(faces.shape(0));
            std::memcpy(faces_vec.data(), faces.data(), sizeof(Face) * faces.shape(0));

            return Mesh(std::move(verts_vec), std::move(faces_vec));
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
        .def("save_preview", &Mesh::save_preview, nb::arg("filename"), nb::arg("width") = 512, nb::arg("height") = 512)
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
        .def("save_as_obj", [](BVHData& self, const char *filename, int max_depth) {
            self.save_as_obj(filename, max_depth);
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
    ;

    nb::class_<CPUBuilder>(m, "CPUBuilder")
        .def(nb::init<const Mesh&>())
        .def("build_bvh", &CPUBuilder::build_bvh, "Use mesh provided in constructor to build BVH with given depth. Returns BVHData.")
    ;

    nb::class_<CPUTraverser>(m, "CPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("traverse", [](
            CPUTraverser& self,
            h_float3_batch& i_ray_origs,
            h_float3_batch& i_ray_vecs,
            h_bool_batch& o_mask,
            h_float_batch& o_t1,
            h_float_batch& o_t2,
            h_uint_batch& o_prim_idx,
            h_float3_batch& o_normals
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.traverse(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t1.data(),
                o_t2.data(),
                o_prim_idx.data(),
                (glm::vec3 *) o_normals.data(),
                n_rays
            );
        })
    ;

    #ifdef CUDA_ENABLED
    nb::enum_<MeshSamplerMode>(m, "MeshSamplerMode")
        .value("SURFACE_UNIFORM", MeshSamplerMode::SURFACE_UNIFORM)
        .export_values()
    ;

    nb::class_<GPUMeshSampler>(m, "GPUMeshSampler")
        .def(nb::init<const Mesh&, MeshSamplerMode, int>(), nb::arg("mesh"), nb::arg("mode"), nb::arg("max_points"))
        .def("sample", [](GPUMeshSampler& self, d_float3_batch& o_points, int n_points) {
            if (n_points > self.max_points_) {
                throw std::runtime_error("n_points exceeds max_points set in constructor");
            }
            if (o_points.shape(0) < n_points || o_points.shape(1) != 3) {
                throw std::runtime_error("o_points has incorrect shape");
            }

            self.sample((glm::vec3 *) o_points.data(), n_points);
        })
    ;

    nb::class_<GPUTraverser>(m, "GPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("traverse", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_bool_batch& o_mask,
            d_float_batch& o_t1,
            d_float_batch& o_t2,
            d_uint_batch& o_prim_idx,
            d_float3_batch& o_normals
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.traverse(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t1.data(),
                o_t2.data(),
                o_prim_idx.data(),
                (glm::vec3 *) o_normals.data(),             
                n_rays
            );
        })
    ;
    #endif
}
