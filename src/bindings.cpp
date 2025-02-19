#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

#include <tuple>

#include "bvh.h"

namespace py = pybind11;

void throw_if_false(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

PYBIND11_MODULE(bvh, m) {
    py::class_<BVH>(m, "BVH")
        .def(py::init<>())
        .def("load_scene", &BVH::load_scene)
        .def("split_faces", &BVH::split_faces)
        .def("memory_bytes", &BVH::memory_bytes)
        .def("build_bvh", &BVH::build_bvh)
        .def("save_as_obj", &BVH::save_as_obj)
        .def("closest_primitive", [](
            BVH& self,
            py::array_t<float> ray_origins,
            py::array_t<float> ray_vectors
        ) {
            auto ray_origins_buf = ray_origins.request();
            auto ray_vectors_buf = ray_vectors.request();

            int n_rays = ray_origins_buf.shape[0];

            throw_if_false(ray_vectors_buf.shape[0] == n_rays, "Mismatched shapes!");
            throw_if_false(ray_origins_buf.ndim == 2 && ray_origins_buf.shape[1] == 3, "ray_origins must have shape (N,3)");
            throw_if_false(ray_origins_buf.itemsize == sizeof(float), "ray_origins must have dtype float32");
            throw_if_false(ray_vectors_buf.ndim == 2 && ray_vectors_buf.shape[1] == 3, "ray_vectors must have shape (N,3)");
            throw_if_false(ray_vectors_buf.itemsize == sizeof(float), "ray_vectors must have dtype float32");

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.request().ptr;
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.request().ptr;

            py::array_t<bool> mask({n_rays});
            py::array_t<float> t({n_rays});

            bool *mask_ptr = (bool *) mask.request().ptr;
            float *t_ptr = (float *) t.request().ptr;

            self.closest_primitive_batch(ray_origins_ptr, ray_vectors_ptr, mask_ptr, t_ptr, n_rays);

            return std::make_tuple(mask, t);
        })
        .def("closest_bbox", [](
            BVH& self,
            py::array_t<float> ray_origins,
            py::array_t<float> ray_vectors
        ) {
            auto ray_origins_buf = ray_origins.request();
            auto ray_vectors_buf = ray_vectors.request();

            int n_rays = ray_origins_buf.shape[0];

            throw_if_false(ray_vectors_buf.shape[0] == n_rays, "Mismatched shapes!");
            throw_if_false(ray_origins_buf.ndim == 2 && ray_origins_buf.shape[1] == 3, "ray_origins must have shape (N,3)");
            throw_if_false(ray_origins_buf.itemsize == sizeof(float), "ray_origins must have dtype float32");
            throw_if_false(ray_vectors_buf.ndim == 2 && ray_vectors_buf.shape[1] == 3, "ray_vectors must have shape (N,3)");
            throw_if_false(ray_vectors_buf.itemsize == sizeof(float), "ray_vectors must have dtype float32");

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.request().ptr;
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.request().ptr;

            py::array_t<bool> mask({n_rays});
            py::array_t<uint32_t> node_idxs({n_rays});
            py::array_t<float> t1({n_rays}), t2({n_rays});

            bool *mask_ptr = (bool *) mask.request().ptr;
            uint32_t *node_idxs_ptr = (uint32_t *) node_idxs.request().ptr;
            float *t1_ptr = (float *) t1.request().ptr;
            float *t2_ptr = (float *) t2.request().ptr;

            self.closest_bbox_batch(ray_origins_ptr, ray_vectors_ptr, mask_ptr, node_idxs_ptr, t1_ptr, t2_ptr, n_rays);

            return std::make_tuple(mask, node_idxs, t1, t2);
        })
        .def("reset_stack", &BVH::reset_stack_batch)
        .def("another_bbox", [](
            BVH& self,
            py::array_t<float> ray_origins,
            py::array_t<float> ray_vectors
        ) {
            auto ray_origins_buf = ray_origins.request();
            auto ray_vectors_buf = ray_vectors.request();

            int n_rays = ray_origins_buf.shape[0];

            throw_if_false(ray_vectors_buf.shape[0] == n_rays, "Mismatched shapes!");
            throw_if_false(ray_origins_buf.ndim == 2 && ray_origins_buf.shape[1] == 3, "ray_origins must have shape (N,3)");
            throw_if_false(ray_origins_buf.itemsize == sizeof(float), "ray_origins must have dtype float32");
            throw_if_false(ray_vectors_buf.ndim == 2 && ray_vectors_buf.shape[1] == 3, "ray_vectors must have shape (N,3)");
            throw_if_false(ray_vectors_buf.itemsize == sizeof(float), "ray_vectors must have dtype float32");

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.request().ptr;
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.request().ptr;

            py::array_t<bool> mask({n_rays});
            py::array_t<uint32_t> node_idxs({n_rays});
            py::array_t<float> t1({n_rays}), t2({n_rays});

            bool *mask_ptr = (bool *) mask.request().ptr;
            uint32_t *node_idxs_ptr = (uint32_t *) node_idxs.request().ptr;
            float *t1_ptr = (float *) t1.request().ptr;
            float *t2_ptr = (float *) t2.request().ptr;

            bool alive = self.another_bbox_batch(ray_origins_ptr, ray_vectors_ptr, mask_ptr, node_idxs_ptr, t1_ptr, t2_ptr, n_rays);

            return std::make_tuple(alive, mask, node_idxs, t1, t2);
        })
        .def("segments", [](
            BVH& self,
            py::array_t<float> ray_origins,
            py::array_t<float> ray_vectors,
            int n_segments
        ) {
            auto ray_origins_buf = ray_origins.request();
            auto ray_vectors_buf = ray_vectors.request();

            int n_rays = ray_origins_buf.shape[0];

            throw_if_false(ray_vectors_buf.shape[0] == n_rays, "Mismatched shapes!");
            throw_if_false(ray_origins_buf.ndim == 2 && ray_origins_buf.shape[1] == 3, "ray_origins must have shape (N,3)");
            throw_if_false(ray_origins_buf.itemsize == sizeof(float), "ray_origins must have dtype float32");
            throw_if_false(ray_vectors_buf.ndim == 2 && ray_vectors_buf.shape[1] == 3, "ray_vectors must have shape (N,3)");
            throw_if_false(ray_vectors_buf.itemsize == sizeof(float), "ray_vectors must have dtype float32");

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.request().ptr;
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.request().ptr;

            py::array_t<bool> segments({n_rays, n_segments});
            bool *segments_ptr = (bool *) segments.request().ptr;

            self.segments_batch(ray_origins_ptr, ray_vectors_ptr, segments_ptr, n_rays, n_segments);

            return segments;
        })
        .def_readonly("depth", &BVH::depth)
        .def_readonly("n_nodes", &BVH::n_nodes)
        .def_readonly("n_leaves", &BVH::n_leaves);
}
