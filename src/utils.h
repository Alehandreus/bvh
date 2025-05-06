#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include <glm/glm.hpp>

#include "cuda_compat.h"

using std::cout, std::endl;

void save_to_bmp(const unsigned int *pixels, int width, int height, const char* filename);

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

struct HitResult {
    bool hit;
    // float t;
    // float t1;
    // float t2;
    union {
        float t;
        struct {
            float t1;
            float t2;
        };
    };
    uint32_t node_idx;
    uint32_t prim_idx;
    glm::vec3 normal;
    glm::vec3 color;

    CUDA_HOST_DEVICE HitResult() : hit(false), t1(0) {}

    CUDA_HOST_DEVICE HitResult(bool hit, float t) : hit(hit), t1(t) {}

    CUDA_HOST_DEVICE HitResult(bool hit, float t1, float t2) : hit(hit), t1(t1), t2(t2) {}
};

struct HitResults {
    bool *masks;
    float *t1;
    float *t2;
    uint32_t *node_idxs;
    glm::vec3 *normals;
    glm::vec3 *colors;

    CUDA_HOST_DEVICE void fill(int i, HitResult hit) {
        masks[i] = hit.hit;
        t1[i] = hit.t1;
        if (t2) t2[i] = hit.t2;        
        node_idxs[i] = hit.node_idx;
        if (normals) normals[i] = hit.normal;
        if (colors) colors[i] = hit.color;
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

int timer(bool start);
void timer_start();
int timer_stop();
