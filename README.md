### BVH Tree
BVH implementation for [neural-intersection](https://github.com/Alehandreus/neural-intersection) repo. Super-early release. Trivial selection criteria, no cuda, no triangle intersection, (almost) nothing.

But GUESS WHAT? PYTHON BINDINGS ARE ALREADY HERE! See `tests/test.py`

## Requirements
* `sudo pacman -S assimp glm openmp` or similar with other package managers
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

## Examples

<img src="https://i.imgur.com/yh6rj9C.png" alt="no image?" style="width: 100%;"/>

<div style="display: flex; justify-content: space-between">
    <img src="https://i.imgur.com/HoO3BY0.png" alt="no image?" style="width: 49%;"/>
    <img src="https://i.imgur.com/sVzMaJX.png" alt="no image?" style="width: 49%;"/>
</div>

## References

* [jbikker's guide to BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [ray-tracing.ru guide](http://ray-tracing.ru/articles184.html)