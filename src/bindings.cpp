#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cpu_traverse.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include "gpu_traverse.cuh"
#endif

namespace nb = nanobind;

using Vector3f = nb::ndarray<float, nb::numpy, nb::shape<3>>;

#ifdef CUDA_ENABLED
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
#endif

NB_MODULE(bvh_impl, m) {
    nb::class_<Mesh>(m, "Mesh")
        .def(nb::init<const char *>())
        .def("split_faces", &Mesh::split_faces)
        .def("bounds", [](Mesh& self) {
            auto [min, max] = self.bounds();
            return nb::make_tuple(
                Vector3f(&min).cast(),
                Vector3f(&max).cast()
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
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_vectors
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            bool *mask_ptr = new bool[n_rays];
            float *t_ptr = new float[n_rays];

            self.closest_primitive(ray_origins_ptr, ray_vectors_ptr, mask_ptr, t_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::numpy>(mask_ptr, {n_rays});
            auto t = nb::ndarray<float, nb::numpy>(t_ptr, {n_rays});

            return nb::make_tuple(mask, t);
        })
         .def("closest_bbox", [](
            CPUTraverser& self,
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

            self.closest_bbox(ray_origins_ptr, ray_vectors_ptr, mask_ptr, node_idxs_ptr, t1_ptr, t2_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::numpy>(mask_ptr, {n_rays});
            auto node_idxs = nb::ndarray<uint32_t, nb::numpy>(node_idxs_ptr, {n_rays});
            auto t1 = nb::ndarray<float, nb::numpy>(t1_ptr, {n_rays});
            auto t2 = nb::ndarray<float, nb::numpy>(t2_ptr, {n_rays});

            return nb::make_tuple(mask, node_idxs, t1, t2);
        })
        .def("another_bbox", [](
            CPUTraverser& self,
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

            bool alive = self.another_bbox(ray_origins_ptr, ray_vectors_ptr, mask_ptr, node_idxs_ptr, t1_ptr, t2_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::numpy>(mask_ptr, {n_rays});
            auto node_idxs = nb::ndarray<uint32_t, nb::numpy>(node_idxs_ptr, {n_rays});
            auto t1 = nb::ndarray<float, nb::numpy>(t1_ptr, {n_rays});
            auto t2 = nb::ndarray<float, nb::numpy>(t2_ptr, {n_rays});

            return nb::make_tuple(alive, mask, node_idxs, t1, t2);
        })
        .def("segments", [](
            CPUTraverser& self,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_origins,
            nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>& ray_vectors,
            int n_segments
        ) {
            uint32_t n_rays = ray_origins.shape(0);

            glm::vec3 *ray_origins_ptr = (glm::vec3 *) ray_origins.data();
            glm::vec3 *ray_vectors_ptr = (glm::vec3 *) ray_vectors.data();

            bool *segments_ptr = new bool[n_rays * n_segments];

            self.segments(ray_origins_ptr, ray_vectors_ptr, segments_ptr, n_rays, n_segments);

            auto segments = nb::ndarray<bool, nb::numpy>(segments_ptr, {n_rays, (uint32_t) n_segments});

            return segments;
        })
    ;

    #ifdef CUDA_ENABLED
    nb::class_<GPUTraverser>(m, "GPUTraverser")
        .def(nb::init<const BVHData&>())
        .def("reset_stack", &GPUTraverser::reset_stack)
        .def("init_rand_state", &GPUTraverser::init_rand_state)
        .def("grow_nbvh", &GPUTraverser::grow_nbvh)
        .def("bbox_raygen", [](GPUTraverser& self, int n_rays_) {
            uint32_t n_rays = n_rays_;

            struct Temp {
                glm::vec3 *ray_origin_ptr;
                glm::vec3 *ray_end_ptr;
                bool *mask_ptr;
                float *t_ptr;
                Temp(int n_rays) {
                    cudaMalloc(&ray_origin_ptr, sizeof(glm::vec3) * n_rays);
                    cudaMalloc(&ray_end_ptr, sizeof(glm::vec3) * n_rays);
                    cudaMalloc(&mask_ptr, sizeof(bool) * n_rays);
                    cudaMalloc(&t_ptr, sizeof(float) * n_rays);
                }
                ~Temp() {
                    cudaFree(ray_origin_ptr);
                    cudaFree(ray_end_ptr);
                    cudaFree(mask_ptr);
                    cudaFree(t_ptr);
                }
            };

            Temp *temp = new Temp(n_rays);

            nb::capsule deleter(temp, [](void *p) noexcept {
                delete (Temp *) p;
            });

            self.bbox_raygen(temp->ray_origin_ptr, temp->ray_end_ptr, temp->mask_ptr, temp->t_ptr, n_rays);

            auto ray_origin = nb::ndarray<float, nb::pytorch, nb::device::cuda>(temp->ray_origin_ptr, {n_rays, 3}, deleter);
            auto ray_end = nb::ndarray<float, nb::pytorch, nb::device::cuda>(temp->ray_end_ptr, {n_rays, 3}, deleter);
            auto mask = nb::ndarray<bool, nb::pytorch, nb::device::cuda>(temp->mask_ptr, {n_rays}, deleter);
            auto t = nb::ndarray<float, nb::pytorch, nb::device::cuda>(temp->t_ptr, {n_rays}, deleter);

            return nb::make_tuple(ray_origin, ray_end, mask, t);
        })
        .def("closest_primitive", [](
            GPUTraverser& self,
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

            self.closest_primitive(ray_origins_ptr, ray_vectors_ptr, temp->mask_ptr, temp->t_ptr, n_rays);

            auto mask = nb::ndarray<bool, nb::pytorch, nb::device::cuda>(temp->mask_ptr, {n_rays}, deleter);
            auto t = nb::ndarray<float, nb::pytorch, nb::device::cuda>(temp->t_ptr, {n_rays}, deleter);

            return nb::make_tuple(mask, t);
        })
        .def("another_bbox", [](
            GPUTraverser& self,
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

            bool alive = self.another_bbox(ray_origins_ptr, ray_vectors_ptr, temp->mask_ptr, temp->node_idxs_ptr, temp->t1_ptr, temp->t2_ptr, n_rays);            

            auto mask = nb::ndarray<bool, nb::pytorch, nb::device::cuda>(temp->mask_ptr, {n_rays}, deleter);
            auto node_idxs = nb::ndarray<uint32_t, nb::pytorch, nb::device::cuda>(temp->node_idxs_ptr, {n_rays}, deleter);
            auto t1 = nb::ndarray<float, nb::pytorch, nb::device::cuda>(temp->t1_ptr, {n_rays}, deleter);
            auto t2 = nb::ndarray<float, nb::pytorch, nb::device::cuda>(temp->t2_ptr, {n_rays}, deleter);

            return nb::make_tuple(alive, mask, node_idxs, t1, t2);
        })
        .def("segments", [](
            GPUTraverser& self,
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

            self.segments(ray_origins_ptr, ray_vectors_ptr, temp->segments_ptr, n_rays, n_segments);

            auto segments = nb::ndarray<bool, nb::pytorch, nb::device::cuda>(temp->segments_ptr, {n_rays, (uint32_t) n_segments}, deleter);

            return segments;
        })
    ;
    #endif
}
