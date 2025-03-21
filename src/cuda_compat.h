#ifdef __CUDACC__
  #define CUDA_ENABLED
  #define CUDA_HOST        __host__
  #define CUDA_DEVICE      __device__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_GLOBAL      __global__
#else
  #define CUDA_HOST
  #define CUDA_DEVICE
  #define CUDA_HOST_DEVICE
  #define CUDA_GLOBAL
#endif

#define CUDA_ENABLED