#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

#include <tuple>

#include "bvh.h"

namespace py = pybind11;

PYBIND11_MODULE(bvh, m) {
    py::class_<BVH>(m, "BVH")
        .def(py::init<>())
        .def("load_scene", &BVH::load_scene)
        .def("split_faces", &BVH::split_faces)
        .def("memory_bytes", &BVH::memory_bytes)
        .def("build_bvh", &BVH::build_bvh)
        .def("save_as_obj", &BVH::save_as_obj)
        .def("intersect_leaves", [](
            BVH& self,
            py::array_t<float> ray_origins,
            py::array_t<float> ray_vectors,
            py::array_t<int> stack_size,
            py::array_t<uint32_t> stack
        ) {
            auto ray_origins_buf = ray_origins.request();
            auto ray_vectors_buf = ray_vectors.request();
            auto stack_size_buf = stack_size.request();
            auto stack_buf = stack.request();

            int n_rays = ray_origins_buf.shape[0];
    
            if (ray_vectors_buf.shape[0] != n_rays || stack_size_buf.shape[0] != n_rays) {
                throw std::runtime_error("Mismatched shapes!");
            }

            if (ray_vectors_buf.ndim != 2 || ray_vectors_buf.shape[1] != 3) {
                throw std::runtime_error("ray_vectors must have shape (N,3)");
            }
            if (ray_vectors_buf.itemsize != sizeof(float)) {
                throw std::runtime_error("ray_vectors must have dtype float32");
            }

            if (stack_size_buf.ndim != 1) {
                throw std::runtime_error("stack_size must have shape (N,)");
            }
            if (stack_size_buf.itemsize != sizeof(int)) {
                throw std::runtime_error("stack_size must have dtype int32");
            }

            if (stack_buf.ndim != 2) {
                throw std::runtime_error("stack must have shape (N,stack_size)");
            }
            if (stack_buf.itemsize != sizeof(uint32_t)) {
                throw std::runtime_error("stack must have dtype uint32");
            }

            int given_stack_size = stack_buf.shape[1];
            if (given_stack_size < self.depth) {
                throw std::runtime_error("Stack size too small!");
            }
            if (ray_origins_buf.ndim != 2 || ray_origins_buf.shape[1] != 3) {
                throw std::runtime_error("ray_origins must have shape (N,3)");
            }
            if (ray_origins_buf.itemsize != sizeof(float)) {
                throw std::runtime_error("ray_origins must have dtype float32");
            }

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.request().ptr;
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.request().ptr;
            int *stack_size_ptr = (int *) stack_size.request().ptr;
            uint32_t *stack_ptr = (uint32_t *) stack.request().ptr;

            py::array_t<bool> mask({n_rays});
            py::array_t<int> leaf_indices({n_rays});
            py::array_t<float> t_enters({n_rays});
            py::array_t<float> t_exits({n_rays});

            bool *mask_ptr = (bool *) mask.request().ptr;
            int *leaf_indices_ptr = (int *) leaf_indices.request().ptr;
            float *t_enters_ptr = (float *) t_enters.request().ptr;
            float *t_exits_ptr = (float *) t_exits.request().ptr;

            #pragma omp parallel for
            for (int i = 0; i < n_rays; ++i) {
                auto [mask, leaf_index, t_enter, t_exit] = self.intersect_leaves(
                    ray_origins_ptr[i],
                    ray_vectors_ptr[i], 
                    stack_size_ptr[i],
                    stack_ptr + given_stack_size * i
                );

                mask_ptr[i] = mask;
                leaf_indices_ptr[i] = leaf_index;
                t_enters_ptr[i] = t_enter;
                t_exits_ptr[i] = t_exit;
            }

            return std::make_tuple(mask, leaf_indices, t_enters, t_exits);
        })
        .def("intersect_segments", [](
            BVH& self,
            py::array_t<float> points_start,
            py::array_t<float> points_end,
            int n_segments
        ) {
            // experiment for Transformer Model at github.com/Alehandreus/neural-intersection

            auto points_start_buf = points_start.request();
            auto points_end_buf = points_end.request();

            int n_rays = points_start_buf.shape[0];

            if (points_end_buf.shape[0] != n_rays) {
                throw std::runtime_error("Mismatched shapes!");
            }

            if (points_start_buf.ndim != 2 || points_start_buf.shape[1] != 3) {
                throw std::runtime_error("points_start must have shape (N,3)");
            }

            if (points_start_buf.itemsize != sizeof(float)) {
                throw std::runtime_error("points_start must have dtype float32");
            }

            if (points_end_buf.ndim != 2 || points_end_buf.shape[1] != 3) {
                throw std::runtime_error("points_end must have shape (N,3)");
            }

            if (points_end_buf.itemsize != sizeof(float)) {
                throw std::runtime_error("points_end must have dtype float32");
            }

            glm::vec3 *points_start_ptr = (glm::vec3 *) points_start.request().ptr;
            glm::vec3 *points_end_ptr = (glm::vec3 *) points_end.request().ptr;

            py::array_t<bool> segments({n_rays, n_segments});
            bool *segments_ptr = (bool *) segments.request().ptr;

            #pragma omp parallel for
            for (int i = 0; i < n_rays; ++i) {
                self.intersect_segments(points_start_ptr[i], points_end_ptr[i], n_segments, segments_ptr + n_segments * i);
            }

            return segments;            
        })
        .def("intersect_primitives", [](
            BVH& self,
            py::array_t<float> points_start,
            py::array_t<float> points_end
        ) {
            // experiment for Transformer Model at github.com/Alehandreus/neural-intersection

            auto points_start_buf = points_start.request();
            auto points_end_buf = points_end.request();

            int n_rays = points_start_buf.shape[0];

            glm::vec3 *points_start_ptr = (glm::vec3 *) points_start.request().ptr;
            glm::vec3 *points_end_ptr = (glm::vec3 *) points_end.request().ptr;

            py::array_t<bool> mask({n_rays});
            py::array_t<float> t({n_rays});

            bool *mask_ptr = (bool *) mask.request().ptr;
            float *t_ptr = (float *) t.request().ptr;

            #pragma omp parallel for
            for (int i = 0; i < n_rays; ++i) {
                auto [a, b] = self.intersect_primitives(points_start_ptr[i], points_end_ptr[i]);
                mask_ptr[i] = a;
                t_ptr[i] = b;
            }

            return std::make_tuple(mask, t);
        })
        .def_readonly("depth", &BVH::depth)
        .def_readonly("n_nodes", &BVH::n_nodes)
        .def_readonly("n_leaves", &BVH::n_leaves);
}
