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
        .def("traverse_primitives", [](
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

            self.traverse_primitives_batch(ray_origins_ptr, ray_vectors_ptr, mask_ptr, t_ptr, n_rays);

            return std::make_tuple(mask, t);
        })
        .def_readonly("depth", &BVH::depth)
        .def_readonly("n_nodes", &BVH::n_nodes)
        .def_readonly("n_leaves", &BVH::n_leaves);
}
