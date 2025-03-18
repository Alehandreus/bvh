#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cpu_traverse.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include "gpu_traverse.cuh"
#endif

namespace nb = nanobind;

using h_float3 = nb::ndarray<float, nb::device::cpu, nb::shape<3>>;

using h_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint32_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig>;

using d_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>;
using d_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_uint32_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig>;

NB_MODULE(bvh_impl, m) {
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
        .def("save_as_obj", [](BVHData& self, const char *filename) {
            self.save_as_obj(filename);
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
        .def("build_bvh", &CPUBuilder::build_bvh)
    ;

    nb::class_<CPUTraverser>(m, "CPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("reset_stack", &CPUTraverser::reset_stack)
        .def("closest_primitive", [](
            CPUTraverser& self,
            h_float3_batch& i_ray_origs,
            h_float3_batch& i_ray_vecs,
            h_bool_batch& o_mask,
            h_float_batch& o_t
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.closest_primitive(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t.data(),
                n_rays
            );
        })
         .def("closest_bbox", [](
            CPUTraverser& self,
            h_float3_batch& i_ray_origs,
            h_float3_batch& i_ray_vecs,
            h_bool_batch& o_mask,
            h_uint32_batch& o_node_idxs,
            h_float_batch& o_t1,
            h_float_batch& o_t2
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.closest_bbox(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_node_idxs.data(),
                o_t1.data(),
                o_t2.data(),
                n_rays
            );
        })
        .def("another_bbox", [](
            CPUTraverser& self,
            h_float3_batch& i_ray_origs,
            h_float3_batch& i_ray_vecs,
            h_bool_batch& o_mask,
            h_uint32_batch& o_node_idxs,
            h_float_batch& o_t1,
            h_float_batch& o_t2
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.another_bbox(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_node_idxs.data(),
                o_t1.data(),
                o_t2.data(),
                n_rays
            );
        })
    ;

    #ifdef CUDA_ENABLED
    nb::class_<GPUTraverser>(m, "GPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("reset_stack", &GPUTraverser::reset_stack)
        .def("init_rand_state", &GPUTraverser::init_rand_state)
        // .def("grow_nbvh", &GPUTraverser::grow_nbvh)
        .def("grow_nbvh", [](GPUTraverser& self, int steps) {
            self.grow_nbvh(steps);
        })
        .def("grow_nbvh", [](GPUTraverser& self) {
            self.grow_nbvh();
        })
        .def("bbox_raygen", [](
            GPUTraverser& self, 
            uint32_t n_rays,
            d_float3_batch& o_ray_origs,
            d_float3_batch& o_ray_vecs,
            d_uint32_batch& o_bbox_idxs,
            d_bool_batch& o_mask,
            d_float_batch& o_t
        ) {
            self.bbox_raygen(
                (glm::vec3 *) o_ray_origs.data(),
                (glm::vec3 *) o_ray_vecs.data(),
                o_bbox_idxs.data(),
                o_mask.data(),
                o_t.data(),
                n_rays
            );
        })
        .def("closest_primitive", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_bool_batch& o_mask,
            d_float_batch& o_t
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.closest_primitive(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t.data(),
                n_rays
            );
        })
        .def("another_bbox", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_bool_batch& o_mask,
            d_uint32_batch& o_node_idxs,
            d_float_batch& o_t1,
            d_float_batch& o_t2,
            bool nbvh_only
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.another_bbox(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_node_idxs.data(),
                o_t1.data(),
                o_t2.data(),
                n_rays,
                nbvh_only
            );
        })
    ;
    #endif
}
