# Set the minimum CMake version and policies for highest tested version
cmake_minimum_required(VERSION 3.15)

# Set up the project and ensure there is a working C++ compiler
# project(bvh LANGUAGES CXX)
project(bvh LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")

set(CMAKE_CUDA_ARCHITECTURES 75)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
    message(WARNING "\
    This CMake file is meant to be executed using 'scikit-build-core'.
    Running it directly will almost certainly not produce the desired
    result. If you are a user trying to install this package, use the
    command below, which will install all necessary build dependencies,
    compile the package in an isolated environment, and then install it.
    =====================================================================
    $ pip install .
    =====================================================================
    If you are a software developer, and this is your own package, then
    it is usually much more efficient to install the build dependencies
    in your environment once and use the following command that avoids
    a costly creation of a new virtual environment at every compilation:
    =====================================================================
    $ pip install nanobind scikit-build-core[pyproject]
    $ pip install --no-build-isolation -ve .
    =====================================================================
    You may optionally add -Ceditable.rebuild=true to auto-rebuild when
    the package is imported. Otherwise, you need to rerun the above
    after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

find_package(assimp REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
    # Name of the extension
    bvh_impl

    # Target the stable ABI for Python 3.12+, which reduces
    # the number of binary wheels that must be built. This
    # does nothing on older Python versions
    STABLE_ABI

    # Source code goes here
    src/bindings.cu
    src/bvh.cu
    src/utils.cu
)

set_target_properties(bvh_impl PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(bvh_impl PRIVATE assimp::assimp)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -x cu --relocatable-device-code=true -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
# target_compile_options(bvh_impl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)

# Install directive for scikit-build-core
install(TARGETS bvh_impl LIBRARY DESTINATION bvh)