#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include <glm/glm.hpp>

#include "utils.h"
#include "cuda_compat.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::cout, std::endl;

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
CUDA_HOST_DEVICE HitResult ray_triangle_intersection(
    const Ray &ray,
    const Face& face,
    const glm::vec3 *vertices,
    bool allow_negative
) {
    float epsilon = 1e-6;

    const glm::vec3 &a = vertices[face.v1];
    const glm::vec3 &b = vertices[face.v2];
    const glm::vec3 &c = vertices[face.v3];

    glm::vec3 edge1 = b - a;
    glm::vec3 edge2 = c - a;
    glm::vec3 ray_cross_e2 = glm::cross(ray.vector, edge2);
    float det = glm::dot(edge1, ray_cross_e2);
    if (det > -epsilon && det < epsilon) {
        return { false, 0 }; // This ray is parallel to this triangle.
    }        

    float inv_det = 1.0 / det;
    glm::vec3 s = ray.origin - a;
    float u = inv_det * glm::dot(s, ray_cross_e2);
    if ((u < 0 && std::fabs(u) > epsilon) || (u > 1 && std::fabs(u-1) > epsilon)) {
        return { false, 0 };
    }

    glm::vec3 s_cross_e1 = glm::cross(s, edge1);
    float v = inv_det * glm::dot(ray.vector, s_cross_e1);
    if ((v < 0 && std::fabs(v) > epsilon) || (u + v > 1 && std::fabs(u + v - 1) > epsilon)) {
        return { false, 0 };
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = inv_det * glm::dot(edge2, s_cross_e1);
    if ((t < epsilon) && (!allow_negative)) {
        return { false, 0 };
    }

    return { true, t };
}

CUDA_HOST_DEVICE glm::vec3 ray_triangle_norm(
    const Face &face,
    const glm::vec3 *vertices
) {
    glm::vec3 n = glm::cross(
        vertices[face.v2] - vertices[face.v1],
        vertices[face.v3] - vertices[face.v1]
    );
    n = glm::normalize(n);

    return n;
}

CUDA_HOST_DEVICE HitResult ray_box_intersection(
    const Ray &ray,
    const BBox &bbox,
    bool allow_negative
) {
    const float eps = 1e-6;

    Ray ray2 = ray;
    if (fabs(ray2.vector.x) < eps) {
        if (ray2.origin.x < bbox.min.x || ray2.origin.x > bbox.max.x) {
            return {false, 0, 0};
        }
        ray2.vector.x = FLT_MAX;
    }
    if (fabs(ray.vector.y) < eps) {
        if (ray2.origin.y < bbox.min.y || ray2.origin.y > bbox.max.y) {
            return {false, 0, 0};
        }
        ray2.vector.y = FLT_MAX;
    }
    if (fabs(ray.vector.z) < eps) {
        if (ray2.origin.z < bbox.min.z || ray2.origin.z > bbox.max.z) {
            return {false, 0, 0};
        }
        ray2.vector.z = FLT_MAX;
    }

    glm::vec3 t1 = (bbox.min - ray2.origin) / ray2.vector;
    glm::vec3 t2 = (bbox.max - ray2.origin) / ray2.vector;

    glm::vec3 tmin = glm::min(t1, t2);
    glm::vec3 tmax = glm::max(t1, t2);

    float t_enter = glm::max(tmin.x, glm::max(tmin.y, tmin.z));
    float t_exit = glm::min(tmax.x, glm::min(tmax.y, tmax.z));

    // Check for valid intersection interval
    if (t_enter > t_exit || (t_exit < 0 && !allow_negative)) {
        return {false, 0, 0};
    }

    return { true, t_enter, t_exit };
}

int timer(bool start) {
    static std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    if (start) {
        start_time = std::chrono::high_resolution_clock::now();
        return 0;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    return duration.count();
}

void timer_start() {
    timer(true);
}

int timer_stop() {
    return timer(false);
}
