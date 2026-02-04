#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include <glm/glm.hpp>

#include "cuda_compat.h"

using std::cout, std::endl;

struct Ray {
    glm::vec3 origin;
    glm::vec3 vector;

    CUDA_HOST_DEVICE Ray(const glm::vec3 &origin, const glm::vec3 &vector) : origin(origin), vector(vector) {}
};

struct Rays {
    glm::vec3 *origs;
    glm::vec3 *vecs;

    CUDA_HOST_DEVICE Ray operator[](int i) const {
        return {origs[i], vecs[i]};
    }

    CUDA_HOST_DEVICE void fill(int i, const Ray &ray) {
        origs[i] = ray.origin;
        vecs[i] = ray.vector;
    }
};

// size_t to uint32_t causes narrowing conversion warning
template <typename T>
uint32_t size(const std::vector<T> &v) {
    return v.size();
}

struct BBox {
    glm::vec3 min, max;

    CUDA_HOST_DEVICE BBox() : min(FLT_MAX), max(-FLT_MAX) {}

    CUDA_HOST_DEVICE BBox(const glm::vec3 &min, const glm::vec3 &max) : min(min), max(max) {}

    CUDA_HOST_DEVICE void update(const glm::vec3 &point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    CUDA_HOST_DEVICE bool inside(const glm::vec3 &point) const {
        return point.x >= min.x && point.x <= max.x &&
                point.y >= min.y && point.y <= max.y &&
                point.z >= min.z && point.z <= max.z;
    }

    CUDA_HOST_DEVICE glm::vec3 diagonal() const {
        return max - min;
    }

    CUDA_HOST_DEVICE float area() const {
        glm::vec3 d = diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    CUDA_HOST_DEVICE BBox get_inflated(float alpha) const {
        return {
            min - alpha * diagonal(),
            max + alpha * diagonal()
        };
    }
};


struct Face {
    uint32_t v1, v2, v3;

    Face() {}

    Face(uint32_t v1, uint32_t v2, uint32_t v3) : v1(v1), v2(v2), v3(v3) {}

    glm::vec3 get_centroid(const glm::vec3 *vertices) const {
        return (vertices[v1] + vertices[v2] + vertices[v3]) / 3.0f;
    }

    BBox get_bounds(const glm::vec3 *vertices) const {
        glm::vec3 min = vertices[v1];
        glm::vec3 max = vertices[v1];

        for (int i = 1; i < 3; i++) {
            min = glm::min(min, vertices[operator[](i)]);
            max = glm::max(max, vertices[operator[](i)]);
        }

        return {min, max};
    }

    float extent(const glm::vec3 *vertices) const {
        return glm::length(get_bounds(vertices).diagonal());
    }

    uint32_t operator[](uint32_t i) const {
        switch (i) {
            case 0: return v1;
            case 1: return v2;
            case 2: return v3;
            default: return 0;
        }
    }
};

struct SDFHitResult {
    float t;
    glm::vec3 closest;
    glm::vec3 barycentrics;
    uint32_t face_idx;
};

struct SDFHitResults {
    float *t;
    glm::vec3 *closests;
    glm::vec3 *barycentricses;
    uint32_t *face_idxs;

    CUDA_HOST_DEVICE void fill(int i, const SDFHitResult &hit) {
        if (t) t[i] = hit.t;       
        if (closests) closests[i] = hit.closest;
        if (barycentricses) barycentricses[i] = hit.barycentrics;
        if (face_idxs) face_idxs[i] = hit.face_idx;
    }
};

struct HitResult {
    bool hit;
    float t;
    uint32_t prim_idx;
    glm::vec3 normal;
    float bary_u, bary_v;
    glm::vec2 uv;

    CUDA_HOST_DEVICE HitResult() : hit(false), t(0), bary_u(0), bary_v(0), uv(0) {}

    CUDA_HOST_DEVICE HitResult(bool hit, float t) : hit(hit), t(t), bary_u(0), bary_v(0), uv(0) {}

    CUDA_HOST_DEVICE HitResult(bool hit, float t, float bary_u, float bary_v) : hit(hit), t(t), bary_u(bary_u), bary_v(bary_v), uv(0) {}
};

struct HitResults {
    bool *masks;
    float *t;
    uint32_t *prim_idxs;
    glm::vec3 *normals;
    glm::vec2 *uvs;

    CUDA_HOST_DEVICE void fill(int i, HitResult hit) {
        if (masks) masks[i] = hit.hit;
        if (t) t[i] = hit.t;
        if (prim_idxs) prim_idxs[i] = hit.prim_idx;
        if (normals) normals[i] = hit.normal;
        if (uvs) uvs[i] = hit.uv;
    }
};

CUDA_HOST_DEVICE HitResult ray_triangle_intersection(
    const Ray &ray,
    const Face& face,
    const glm::vec3 *vertices,
    bool allow_negative = false
);

CUDA_HOST_DEVICE glm::vec3 ray_triangle_norm(
    const Face &face,
    const glm::vec3 *vertices
);

CUDA_HOST_DEVICE HitResult ray_box_intersection(
    const Ray &ray,
    const BBox &bbox,
    bool allow_negative = false
);

CUDA_HOST_DEVICE float box_df(
    const glm::vec3& p,
    const BBox& box
);

CUDA_HOST_DEVICE SDFHitResult triangle_sdf(
    const glm::vec3 &p,
    const Face& face,
    const glm::vec3 *vertices
);

CUDA_HOST_DEVICE inline float vdot(const glm::vec3 &a, const glm::vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

int timer(bool start);
void timer_start();
int timer_stop();
