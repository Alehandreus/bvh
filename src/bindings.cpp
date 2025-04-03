#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cpu_traverse.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include "gpu_traverse.cuh"
#endif

namespace nb = nanobind;

using h_float3 = nb::ndarray<float, nb::shape<3>, nb::numpy>;

using h_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
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
    nb::enum_<TreeType>(m, "TreeType")
        .value("BVH", TreeType::BVH)
        .value("NBVH", TreeType::NBVH)
        .export_values()
    ;

    nb::enum_<TraverseMode>(m, "TraverseMode")
        .value("CLOSEST_PRIMITIVE", TraverseMode::CLOSEST_PRIMITIVE)
        .value("CLOSEST_BBOX", TraverseMode::CLOSEST_BBOX)
        .value("ANOTHER_BBOX", TraverseMode::ANOTHER_BBOX)
        .export_values()
    ;

    nb::class_<Mesh>(m, "Mesh")
        .def(nb::init<const char *>())
        .def("split_faces", &Mesh::split_faces)
        .def("bounds", [](Mesh& self) {
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
        .def("reset_stack", &CPUTraverser::reset_stack)
        .def("traverse", [](
            CPUTraverser& self,
            h_float3_batch& i_ray_origs,
            h_float3_batch& i_ray_vecs,
            h_bool_batch& o_mask,
            h_float_batch& o_t1,
            h_float_batch& o_t2,
            h_uint_batch& o_node_idx,
            h_float3_batch& o_normals,
            TreeType tree_type,
            TraverseMode mode
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            bool alive = self.traverse(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t1.data(),
                o_t2.data(),
                o_node_idx.data(),
                (glm::vec3 *) o_normals.data(),
                n_rays,
                tree_type,
                mode
            );

            return alive;
        })
    ;

    #ifdef CUDA_ENABLED
    nb::class_<GPUTraverser>(m, "GPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("reset_stack", &GPUTraverser::reset_stack)
        .def("grow_nbvh", &GPUTraverser::grow_nbvh, nb::arg("steps") = 1)
        .def("traverse", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_bool_batch& o_mask,
            d_float_batch& o_t1,
            d_float_batch& o_t2,
            d_uint_batch& o_node_idx,
            d_float3_batch& o_normals,        
            TreeType tree_type,
            TraverseMode mode
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            bool alive = self.traverse(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t1.data(),
                o_t2.data(),
                o_node_idx.data(),
                (glm::vec3 *) o_normals.data(),             
                n_rays,
                tree_type,
                mode
            );

            return alive;
        })
        .def("fill_history", [](
            GPUTraverser& self,
            d_bool_batch& i_masks,
            d_uint_batch& i_node_idxs,
            d_int_batch& o_depths,
            d_uintN_batch& o_history
        ) {
            uint32_t n_rays = i_node_idxs.shape(0);

            self.fill_history(
                i_masks.data(),
                i_node_idxs.data(),
                o_depths.data(),
                o_history.data(),
                n_rays
            );
        })
    ;

    nb::class_<GPURayGen>(m, "GPURayGen")
        .def(nb::init<GPUTraverser&, int>())
        .def("raygen", [](
            GPURayGen& self,
            d_float3_batch& o_ray_origs,
            d_float3_batch& o_ray_vecs,
            d_bool_batch& o_masks,
            d_float_batch& o_t1,
            d_uint_batch& o_bbox_idxs,
            d_float3_batch& o_normals
        ) {
            int n_generated = self.raygen(
                (glm::vec3 *) o_ray_origs.data(),
                (glm::vec3 *) o_ray_vecs.data(),
                o_masks.data(),
                o_t1.data(),
                o_bbox_idxs.data(),
                (glm::vec3 *) o_normals.data()
            );

            return n_generated;
        })
    ;
    #endif
}
