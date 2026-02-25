# bvh: GPU-Accelerated Mesh Ray Tracing — Project Reference

## High-Level Overview

This project is a **GPU-accelerated mesh intersection library with Python bindings**. The core idea:

- 3D meshes (OBJ, FBX, GLTF, etc.) are loaded via Assimp and a BVH is built using SAH splitting.
- Ray queries (closest hit, all hits) and signed distance field (SDF/point) queries run on either CPU (OpenMP) or GPU (CUDA).
- A uniform surface sampler generates random points on mesh surfaces.
- Python bindings via nanobind expose a NumPy-compatible batch API.

The library is designed to support neural rendering pipelines where fast, differentiable-friendly mesh queries are needed without a full rendering framework.

---

## Architecture Map

```
[Python User Code]
        ↓
[nanobind bindings.cpp]          # batch NumPy API
        ↓
┌─────────────────────────────┐
│  Mesh (load + BVH build)    │  mesh.h / mesh.cpp / build.h / build.cpp
└─────────────────────────────┘
        ↓
   ┌────┴────┐
   ↓         ↓
CPUTraverser  GPUTraverser      # cpu_traverse.h/cpp  gpu_traverse.cuh/cu
   ↓         ↓
 OpenMP    CUDA kernels
   ↓         ↓
 Shared geometry utilities      # utils.h / utils.cpp  (CUDA_HOST_DEVICE)
   ↓         ↓
 BVH stack traversal → Ray-Triangle / Ray-Box / SDF
                ↓
        HitResult / SDFHitResult
```

---

## Directory Structure

```
bvh/
├── src/                        # All project source code
│   ├── cuda_compat.h           # __host__ / __device__ macros
│   ├── utils.h / utils.cpp     # Core geometry: Ray, BBox, intersection, SDF
│   ├── material.h              # Material + Texture structs
│   ├── texture_sample.h        # CPU bilinear texture sampling
│   ├── texture_loader.h/.cpp   # stb_image loader (PNG/JPG + embedded GLTF)
│   ├── build.h / build.cpp     # BVH construction (SAH via madmann91/bvh)
│   ├── mesh.h / mesh.cpp       # Mesh loading, BVH access, preview rendering
│   ├── cpu_traverse.h/.cpp     # CPU ray/SDF queries (OpenMP)
│   ├── gpu_traverse.cuh/.cu    # GPU ray/SDF queries (CUDA)
│   ├── mesh_sampler.cuh/.cu    # GPU uniform surface sampler
│   ├── bindings.cpp            # nanobind Python module
│   ├── stb_image.h             # Bundled STB image loader
│   └── stb_image_write.h       # Bundled STB image writer
├── bvh/                        # Submodule: madmann91/bvh SAH builder library
└── CMakeLists.txt              # Build system
```

---

## Data Structures

### Geometry (`utils.h`)

```cpp
struct Ray {
    glm::vec3 origin, direction;
    float tmin, tmax;
};

struct Rays {          // batch of N rays
    glm::vec3* origins;
    glm::vec3* directions;
    int n;
};

struct BBox {
    glm::vec3 min, max;
};

struct Face {          // triangle: 3 vertex indices + material index
    int a, b, c, material_id;
};

struct HitResult {
    float t;              // ray parameter at hit
    float u, v;           // barycentric coords
    int face_id;          // index into faces array
    glm::vec3 normal;     // interpolated shading normal
    glm::vec2 uv;         // texture UV
    glm::vec3 color;      // resolved material color (constant or texture)
    bool hit;
};

struct SDFHitResult {
    float dist;           // signed distance (negative = inside)
    float u, v;           // barycentric coords of closest point
    int face_id;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 color;
    bool hit;             // always true if any face exists
};
```

### BVH (`build.h`)

```cpp
struct BVHNode {
    BBox bbox;
    int left_child, right_child;   // -1 for leaves
    int face_start, face_count;    // leaf face range
};

struct BVHData {
    std::vector<BVHNode> nodes;
    std::vector<Face> reordered_faces;   // sorted to BVH leaf order
    int max_depth, leaf_count;
};
```

### Material / Texture (`material.h`)

```cpp
struct Texture {
    std::vector<uint8_t> pixels;   // always RGBA (4-channel)
    int width, height, channels;
};

struct Material {
    glm::vec3 base_color;
    int texture_id;   // -1 = constant color, else index into textures array
};
```

---

## Key Algorithms

### BVH Construction (`build.cpp`)

Uses the `madmann91/bvh` library with a parallel sweep-SAH builder:

1. Compute per-triangle bounding boxes and centroids.
2. Run `bvh::v2::DefaultBuilder` (SAH with parallel executor).
3. Reorder faces to BVH leaf order for coherent memory access.
4. Output flat `BVHNode[]` array usable on both CPU and GPU.

`BVHData::save_to_obj()` exports each leaf AABB as an OBJ wireframe box for visualization.

### Ray-Triangle Intersection (`utils.cpp`)

Möller–Trumbore algorithm:
- Computes edge vectors and determinant; rejects parallel rays.
- Returns barycentric coordinates `(u, v)` and ray parameter `t`.
- Epsilon guard: `t > 1e-6` to avoid self-intersection.

### Ray-AABB Intersection (`utils.cpp`)

Slab method:
- Intersects ray with 3 axis-aligned slab pairs.
- Takes max of entry slabs, min of exit slabs.
- `allow_negative` flag enables backward intersection for shell tracing.

### Point-to-Triangle SDF (`utils.cpp`)

Ericson's closest-point algorithm (7-region Voronoi decomposition):
- Projects point onto triangle plane; tests vertex/edge/face regions.
- Returns signed distance (negative = inside the mesh, based on normal direction).
- Handles degenerate triangles by returning large positive distance.

### BVH Stack Traversal (shared CPU/GPU)

Used in both `cpu_traverse.cpp` and `gpu_traverse.cu` via a common inlined function:
- Iterative DFS with explicit stack (no recursion; GPU-safe).
- AABB rejection gate before per-triangle test.
- `allow_forward` / `allow_backward` flags for front/back-face culling.
- Single-hit variant: tracks closest `t` and overwrites on improvement.
- All-hit variant: stores all intersections and sorts by `t` (insertion sort on GPU).

### GPU Texture Sampling (`gpu_traverse.cu`)

Bilinear texture sampling:
- UV coordinates wrapped to `[0, 1]` via `fmod`.
- 4 nearest pixel samples fetched and bilinearly interpolated.
- Applied after BVH traversal to resolve final surface color.

### Uniform Surface Sampling (`mesh_sampler.cu`)

1. Precompute per-face areas; build a CDF over faces (weighted by area).
2. Each CUDA thread draws a random face via binary search on CDF.
3. Uniform barycentric sampling within the chosen triangle: `(sqrt(r1), r2)` transform.

---

## File-by-File Function Reference

### `utils.h / utils.cpp`

- `ray_triangle_intersection(ray, v0, v1, v2)` → `(t, u, v, hit)` — Möller-Trumbore ray-triangle test.
- `ray_triangle_norm(v0, v1, v2)` → `vec3` — Triangle geometric normal; returns zero-safe fallback.
- `ray_box_intersection(ray, bbox, allow_negative)` → `(tmin, tmax, hit)` — Slab AABB intersection.
- `box_df(point, bbox)` → `float` — Distance from point to AABB surface (0 if inside).
- `triangle_sdf(point, v0, v1, v2, n)` → `(dist, u, v)` — Signed distance from point to triangle with closest barycentric coords.

### `build.h / build.cpp`

- `CPUBuilder::build_bvh(vertices, faces)` → `BVHData` — SAH BVH construction via madmann91/bvh.
- `BVHData::save_to_obj(path)` — Write each BVH leaf AABB as an OBJ wireframe for debugging.

### `texture_loader.h / texture_loader.cpp`

- `loadImageFromFile(path)` → `Texture` — Load PNG/JPG from disk via stb_image; convert to RGBA.
- `loadImageFromEmbedded(data, size)` → `Texture` — Decode embedded GLTF texture bytes.

### `texture_sample.h`

- `sampleTextureCPU(texture, uv)` → `vec3` — CPU bilinear texture lookup with UV wrap.
- `srgbToLinear(c)` → `float` — sRGB gamma decode: `c^2.2` approximation.

### `mesh.h / mesh.cpp`

- `Mesh::from_file(path, up_axis, forward_axis)` — Load mesh via Assimp; extract geometry, materials, textures.
- `Mesh::build_bvh_internal()` — Delegate to `CPUBuilder`; store result in `bvh_data_`.
- `Mesh::get_bvh()` — Lazy-build BVH on first call; return `BVHData&`.
- `Mesh::bounds()` → `BBox` — Axis-aligned bounding box of all vertices.
- `Mesh::get_c()` → `vec3` — Bounding sphere center.
- `Mesh::get_R()` → `float` — Bounding sphere radius.
- `Mesh::save_preview(path, width, height)` — Software rasterizer: orthographic projection + depth test → PNG.
- `Mesh::save_to_obj(path)` — Export raw geometry as OBJ.

### `cpu_traverse.h / cpu_traverse.cpp`

- `CPUTraverser::CPUTraverser(mesh)` — Store pointer to mesh; no device allocation.
- `CPUTraverser::ray_query(rays)` → `HitResult[]` — OpenMP-parallel BVH traversal for N rays; returns closest hit per ray.
- `CPUTraverser::point_query(points)` → `SDFHitResult[]` — OpenMP-parallel SDF query for N points.
- `CPUTraverser::generate_camera_rays(width, height)` → `Rays` — Auto-fit perspective camera for the mesh bounds; generate pixel rays.

### `gpu_traverse.cuh / gpu_traverse.cu`

- `GPUTraverser::GPUTraverser(mesh)` — Upload vertices, faces, BVH nodes, materials, textures to device.
- `GPUTraverser::ray_query(rays, n)` → `HitResult[]` (device) — Launch `ray_query_entry` kernel; one thread per ray.
- `GPUTraverser::ray_query_all(rays, n, max_hits)` → `HitResult[]` (device) — All-hit variant; results sorted by `t` per ray.
- `GPUTraverser::point_query(points, n)` → `SDFHitResult[]` (device) — Launch `point_query_entry` kernel.
- `sampleTextureGPU(pixels, w, h, uv)` → `vec3` — Device-side bilinear texture lookup.
- `ray_query_entry(...)` — CUDA kernel: per-ray BVH traversal + texture resolve.
- `point_query_entry(...)` — CUDA kernel: per-point SDF traversal.
- `ray_query_all_gpu(...)` — CUDA kernel: all-hit traversal with insertion sort.
- `ray_query_all_entry(...)` — Kernel launcher wrapper for `ray_query_all_gpu`.

### `mesh_sampler.cuh / mesh_sampler.cu`

- `GPUMeshSampler::GPUMeshSampler(mesh, n_samples)` — Precompute face area CDF; allocate cuRand states; allocate output buffers.
- `GPUMeshSampler::sample()` → `(points, barycentrics, face_ids)` (device) — Launch sampling kernel; returns positions, `(u,v)` barycentrics, and face indices.
- `mesh_sample_surface_uniform_entry(...)` — CUDA kernel: binary search on CDF, uniform barycentric sample, transform to world position.

### `bindings.cpp`

nanobind module `mesh_utils_impl` exposing four classes to Python:

- **`Mesh`**
  - `.from_file(path, up_axis, forward_axis)` — Class method; load from disk.
  - `.vertices` → `ndarray(N,3,f32)` — All vertex positions.
  - `.faces` → `ndarray(M,3,i32)` — Triangle index triplets.
  - `.normals` → `ndarray(N,3,f32)` — Per-vertex normals.
  - `.uvs` → `ndarray(N,2,f32)` — Per-vertex UVs.
  - `.save_preview(path, w, h)` — Render orthographic preview PNG.
  - `.save_to_obj(path)` — Export geometry as OBJ.

- **`CPUTraverser`**
  - `.ray_query(origins, directions)` → dict of arrays — Closest hit for each ray.
  - `.point_query(points)` → dict of arrays — SDF for each point.

- **`GPUTraverser`**
  - `.ray_query(origins, directions)` → dict of arrays — GPU closest-hit batch.
  - `.ray_query_all(origins, directions, max_hits)` → dict of arrays — All hits, sorted.
  - `.point_query(points)` → dict of arrays — GPU SDF batch.

- **`GPUMeshSampler`**
  - `.sample()` → `(points, barycentrics, face_ids)` — Uniform surface sample batch.

All array inputs/outputs are NumPy-compatible; GPU results are copied to CPU before returning.

---

## Build System (`CMakeLists.txt`)

- **Standard:** C++20, CUDA 17.
- **GPU arch:** `sm_120` (Ada Lovelace / RTX 4000 series).
- **Python extension:** `mesh_utils_impl` with nanobind `STABLE_ABI`.
- CPU compilation units: `mesh.cpp`, `build.cpp`, `texture_loader.cpp`, `bvh` library sources, `utils.cpp`, `cpu_traverse.cpp`, `bindings.cpp`.
- CUDA compilation units: `gpu_traverse.cu`, `mesh_sampler.cu`.
- Dependencies: nanobind, assimp.

```bash
pip install .          # installs GPU-compiled Python extension
```

---

## Dependencies

| Library | Role |
|---------|------|
| **madmann91/bvh** | SAH BVH construction (submodule in `bvh/`) |
| **Assimp** | 3D model loading (OBJ, FBX, GLTF, etc.) |
| **GLM** | Vector/matrix math (`glm::vec3`, `glm::vec2`) |
| **stb_image** | PNG/JPG image I/O (bundled in `src/`) |
| **nanobind** | Python C++ bindings |
| **CUDA / cuRand** | GPU computation and random sampling |
| **OpenMP** | CPU parallel traversal |

---

## GPU Memory Layout

Per-mesh device buffers (allocated in `GPUTraverser`):

| Buffer | Type | Contents |
|--------|------|---------|
| `d_vertices` | `vec3[]` | Vertex positions |
| `d_normals` | `vec3[]` | Per-vertex normals |
| `d_uvs` | `vec2[]` | Per-vertex UVs |
| `d_faces` | `Face[]` | Triangle indices + material ID |
| `d_nodes` | `BVHNode[]` | BVH tree nodes (flat array) |
| `d_materials` | `Material[]` | Base colors + texture IDs |
| `d_texture_pixels` | `uint8_t*[]` | Per-texture pixel data pointers |

Per-query output buffers (allocated per `ray_query` / `point_query` call):

| Buffer | Type | Contents |
|--------|------|---------|
| `d_results` | `HitResult[]` | Per-ray hit info |
| `d_sdf_results` | `SDFHitResult[]` | Per-point SDF info |
