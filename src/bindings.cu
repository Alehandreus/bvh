#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "bvh.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace nb = nanobind;

struct HitResultCuda {
    bool *mask_ptr;
    union {
        float *t_ptr;
        struct {
            float *t1_ptr;
            float *t2_ptr;
            uint32_t *node_idxs_ptr;
        };
    };

    HitResultCuda(uint32_t n_rays) {
        cudaMalloc(&mask_ptr, n_rays * sizeof(bool));
        cudaMalloc(&t1_ptr, n_rays * sizeof(float));
        cudaMalloc(&t2_ptr, n_rays * sizeof(float));
        cudaMalloc(&node_idxs_ptr, n_rays * sizeof(uint32_t));
    }

    ~HitResultCuda() {
        cudaFree(mask_ptr);
        cudaFree(t1_ptr);
        cudaFree(t2_ptr);
        cudaFree(node_idxs_ptr);
    }
};

NB_MODULE(bvh_impl, m) {
    nb::class_<BVH>(m, "BVH")
        .def(nb::init<>())
        .def("load_scene", &BVH::load_scene)
        .def("split_faces", &BVH::split_faces)
        .def("memory_bytes", &BVH::memory_bytes)
        .def("build_bvh", &BVH::build_bvh)
        .def("save_as_obj", &BVH::save_as_obj)
        .def("closest_primitive", [](
            BVH& self,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_vectors
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            bool *mask_ptr = new bool[n_rays];
            float *t_ptr = new float[n_rays];

            self.closest_primitive_batch(ray_origins_ptr, ray_vectors_ptr, mask_ptr, t_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::numpy>(mask_ptr, {n_rays});
            auto t = nb::ndarray<float, nb::numpy>(t_ptr, {n_rays});

            return nb::make_tuple(mask, t);
        })
        #ifdef CUDA_ENABLED
        .def("cudify", &BVH::cudify)
        .def("closest_primitive_cuda", [](
            BVH& self,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>& ray_vectors
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            HitResultCuda *temp = new HitResultCuda(n_rays);

            nb::capsule deleter(temp, [](void *p) noexcept {
                delete (HitResultCuda *) p;
            });

            self.closest_primitive_cuda(ray_origins_ptr, ray_vectors_ptr, temp->mask_ptr, temp->t_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::pytorch, nb::device::cuda>(temp->mask_ptr, {n_rays}, deleter);
            auto t = nb::ndarray<float, nb::pytorch, nb::device::cuda>(temp->t_ptr, {n_rays}, deleter);

            return nb::make_tuple(mask, t);
        })
        .def("segments_cuda", [](
            BVH& self,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>& ray_vectors,
            int n_segments
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            struct Temp {
                bool *segments_ptr;
                Temp(uint32_t n_rays, uint32_t n_segments) {
                    cudaError_t err = cudaMalloc(&segments_ptr, n_rays * n_segments * sizeof(bool));
                    if (err != cudaSuccess) {
                        cout << cudaGetErrorString(err) << endl;
                        throw std::runtime_error("cudaMalloc failed");
                    }
                }
                ~Temp() {
                    cudaFree(segments_ptr);
                }
            };

            Temp *temp = new Temp(n_rays, n_segments);

            nb::capsule deleter(temp, [](void *p) noexcept {
                delete (Temp *) p;
            });

            self.segments_cuda(ray_origins_ptr, ray_vectors_ptr, temp->segments_ptr, n_rays, n_segments);

            auto segments = nb::ndarray<bool, nb::pytorch, nb::device::cuda>(temp->segments_ptr, {n_rays, (uint32_t) n_segments}, deleter);

            return segments;
        })
        #endif
        .def("closest_bbox", [](
            BVH& self,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_vectors
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            bool *mask_ptr = new bool[n_rays];
            uint32_t *node_idxs_ptr = new uint32_t[n_rays];
            float *t1_ptr = new float[n_rays];
            float *t2_ptr = new float[n_rays];

            self.closest_bbox_batch(ray_origins_ptr, ray_vectors_ptr, mask_ptr, node_idxs_ptr, t1_ptr, t2_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::numpy>(mask_ptr, {n_rays});
            auto node_idxs = nb::ndarray<uint32_t, nb::numpy>(node_idxs_ptr, {n_rays});
            auto t1 = nb::ndarray<float, nb::numpy>(t1_ptr, {n_rays});
            auto t2 = nb::ndarray<float, nb::numpy>(t2_ptr, {n_rays});

            return nb::make_tuple(mask, node_idxs, t1, t2);
        })
        .def("reset_stack", &BVH::reset_stack_batch)
        .def("another_bbox", [](
            BVH& self,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_vectors
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            bool *mask_ptr = new bool[n_rays];
            uint32_t *node_idxs_ptr = new uint32_t[n_rays];
            float *t1_ptr = new float[n_rays];
            float *t2_ptr = new float[n_rays];

            bool alive = self.another_bbox_batch(ray_origins_ptr, ray_vectors_ptr, mask_ptr, node_idxs_ptr, t1_ptr, t2_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::numpy>(mask_ptr, {n_rays});
            auto node_idxs = nb::ndarray<uint32_t, nb::numpy>(node_idxs_ptr, {n_rays});
            auto t1 = nb::ndarray<float, nb::numpy>(t1_ptr, {n_rays});
            auto t2 = nb::ndarray<float, nb::numpy>(t2_ptr, {n_rays});

            return nb::make_tuple(alive, mask, node_idxs, t1, t2);
        })
        .def("segments", [](
            BVH& self,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_vectors,
            int n_segments
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            bool *segments_ptr = new bool[n_rays * n_segments];

            self.segments_batch(ray_origins_ptr, ray_vectors_ptr, segments_ptr, n_rays, n_segments);

            auto segments = nb::ndarray<bool, nb::numpy>(segments_ptr, {n_rays, (uint32_t) n_segments});

            return segments;
        })
        .def_ro("depth", &BVH::depth)
        .def_ro("n_nodes", &BVH::n_nodes)
        .def_ro("n_leaves", &BVH::n_leaves)
    ;
}
