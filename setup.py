from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "bvh",
        [
            "src/bindings.cpp",
            "src/bvh.cpp",
        ],
        include_dirs=["include"],
        libraries=["assimp"],
        extra_compile_args=["-fopenmp"],
    ),
]

setup(
    name="bvh",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
