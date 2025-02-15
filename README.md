### BVH Tree
BVH implementation for [neural-intersection](https://github.com/Alehandreus/neural-intersection) repo. Super-early release. Trivial selection criteria, no cuda, no triangle intersection, (almost) nothing.

But GUESS WHAT? PYTHON BINDINGS ARE ALREADY HERE! See `tests/test.py`

## Requirements
* `sudo pacman -S assimp glm` or similar with other package managers
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

<div style="display: flex;">
    <img src="https://i.imgur.com/HoO3BY0.png" alt="no image?" style="width: 50%;"/>
    <img src="https://i.imgur.com/sVzMaJX.png" alt="no image?" style="width: 50%;"/>
</div>

