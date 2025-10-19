### BVH Tree
C++ / CUDA implementation of BVH Tree for [mesh-mapping](https://github.com/Alehandreus/neural-intersection) repo.

## Features
* Large primitive splitting
* Uniform sampling on mesh surface
* CUDA traversal
* NumPy / PyTorch bindings

## Requirements
* **assimp**, **glm**, **openmp**
    * `sudo pacman -S assimp glm openmp`
    * `sudo apt install libassimp-dev libglm-dev` (probably)
* **CUDA >=11**
    * `sudo pacman -S cuda`
* `pip install -r requirements.txt`
* [github.com/madmann91/bvh](https://github.com/madmann91/bvh/)

## Installation

Building Python package without GPU support is not implemented. A day may come when the necessary CMake logic is added, but it is not this day. This day (a) replace 120 in `CMakeLists.txt` with [your CC](https://developer.nvidia.com/cuda-gpus) and (b) run
```
pip install .
```

To build and run CPU-only C++ example, run
```
make build_cpu
make run
```

To build and run C++ example with GPU support, run
```
make build_gpu
make run
```

## Usage

See `examples` folder for both Python and C++ code.

You can get `suzanne.fbx` (Blender monkey) from [Google Drive](https://drive.google.com/file/d/1WsVTUILUjK1UWBZuOMLQD-sr-KQHSCRL/view?usp=sharing) 
or with [gdown](https://github.com/wkentaro/gdown) (`pip install gdown`):
```
gdown 1WsVTUILUjK1UWBZuOMLQD-sr-KQHSCRL
```

## Images

<img src="https://i.imgur.com/yh6rj9C.png" alt="no image?" style="width: 100%;"/>
<img src="https://i.imgur.com/sVzMaJX.png" alt="no image?" style="width: 100%;"/>

## References

* [jbikker's guide to BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [ray-tracing.ru guide](http://ray-tracing.ru/articles184.html)