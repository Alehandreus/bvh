### BVH Tree
BVH implementation for [neural-intersection](https://github.com/Alehandreus/neural-intersection) repo. Early release. But GUESS WHAT? PYTHON BINDINGS ARE ALREADY HERE! See `tests/test.py`

## Features
* Large primitive splitting as preprocessing step
* NumPy bindings

## Requirements
* `assimp`, `glm`, `openmp`
    * `sudo pacman -S assimp glm openmp`
    * `sudo apt install libassimp-dev libglm-dev`
* `pip install -r requirements.txt`

## Installation

To install as python package, run
```
pip install .
```

To build and run as C++ program, run
```
make release
make run
```

## Images

<img src="https://i.imgur.com/yh6rj9C.png" alt="no image?" style="width: 100%;"/>
<img src="https://i.imgur.com/sVzMaJX.png" alt="no image?" style="width: 100%;"/>

## References

* [jbikker's guide to BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [ray-tracing.ru guide](http://ray-tracing.ru/articles184.html)