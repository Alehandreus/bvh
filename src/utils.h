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

    CUDA_HOST_DEVICE Ray(const glm::vec3 &origin, const glm::vec3 &vector);
};

// size_t to uint32_t causes narrowing conversion warning
// no templates this time
uint32_t size(const std::vector<float> &v);
uint32_t size(const std::vector<uint32_t> &v);
uint32_t size(const std::vector<glm::vec3> &v);

struct Face {
    uint32_t v1, v2, v3;
    glm::vec3 centroid;

    Face(uint32_t v1, uint32_t v2, uint32_t v3);
    void calc_centroid(const glm::vec3 *vertices);
    float extent(const glm::vec3 *vertices) const;
    uint32_t operator[](uint32_t i) const;
};

struct BBox {
    glm::vec3 min, max;

    CUDA_HOST_DEVICE BBox();
    CUDA_HOST_DEVICE void update(const glm::vec3 &point);
    CUDA_HOST_DEVICE bool inside(const glm::vec3 &point) const;
    CUDA_HOST_DEVICE glm::vec3 diagonal() const;
};

struct HitResult {
    bool hit;
    union {
        float t;
        struct {
            float t1;
            float t2;
            uint32_t node_idx;
        };
    };

    CUDA_HOST_DEVICE HitResult();
    CUDA_HOST_DEVICE HitResult(bool hit, float t);
    CUDA_HOST_DEVICE HitResult(bool hit, float t1, float t2);
};

CUDA_HOST_DEVICE HitResult ray_triangle_intersection(
    const Ray &ray,
    const Face& face,
    const glm::vec3 *vertices
);


CUDA_HOST_DEVICE HitResult ray_box_intersection(
    const Ray &ray,
    const BBox &bbox
);

int timer(bool start);
void timer_start();
int timer_stop();
