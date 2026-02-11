cmake_minimum_required(VERSION 3.15)
project(mesh_utils LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)
find_package(nanobind CONFIG REQUIRED)
find_package(assimp REQUIRED)

include_directories(bvh/src)

# this code requires c++20
set(CPU_SOURCES
    src/mesh.cpp
    src/build.cpp
    src/texture_loader.cpp
    src/texture_sample.cpp
    bvh/src/bvh/v2/c_api/bvh.cpp
)

set(CUDA_SOURCES
    src/utils.cpp
    src/cpu_traverse.cpp
    src/bindings.cpp
    src/gpu_traverse.cu
    src/mesh_sampler.cu
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

nanobind_add_module(mesh_utils_impl STABLE_ABI ${CPU_SOURCES} ${CUDA_SOURCES})

target_compile_options(mesh_utils_impl PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>
)

set_property(TARGET mesh_utils_impl PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET mesh_utils_impl PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(mesh_utils_impl PRIVATE assimp::assimp)
install(TARGETS mesh_utils_impl LIBRARY DESTINATION mesh_utils)
