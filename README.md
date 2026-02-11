### Mesh Utils
A library for ray tracing meshes on GPU with PyTorch bindings.

## Requirements
* **assimp**, **glm**, **openmp**
    * `sudo pacman -S assimp glm openmp`
    * `sudo apt install libassimp-dev libglm-dev` (probably)
* **CUDA >=11**
    * `sudo pacman -S cuda`
* `pip install -r requirements.txt`
* [madmann91/bvh](https://github.com/madmann91/bvh/) builder:
    * Added automatically when cloning with `--recurse-submodules`
    * Can be added to cloned repository with `git submodule update --init`

## Installation

Building Python package without GPU support is not implemented. A day may come when the necessary CMake logic is added, but it is not this day. This day (a) replace 120 in `CMakeLists.txt` with [your CC](https://developer.nvidia.com/cuda-gpus) and (b) run
```
pip install .
```

## Usage

See `examples` directory, in particular `example_gpu.py`.

Get `suzanne.fbx` (Blender monkey) from [Google Drive](https://drive.google.com/file/d/1WsVTUILUjK1UWBZuOMLQD-sr-KQHSCRL/view?usp=sharing) 
or with [gdown](https://github.com/wkentaro/gdown) (`pip install gdown`):
```
gdown 1WsVTUILUjK1UWBZuOMLQD-sr-KQHSCRL
```

Get `shrek.glb` from [Sketchfab](https://sketchfab.com/3d-models/shrek-9aed77fc50814923a734053fbd8d61bf)

## References

* [jbikker's guide to BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [ray-tracing.ru guide](http://ray-tracing.ru/articles184.html)