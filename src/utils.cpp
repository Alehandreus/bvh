#include <vector>
#include <glm/glm.hpp>

#include "utils.h"

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
std::tuple<bool, float> // mask, t
ray_intersects_triangle(
    const glm::vec3 &ray_origin,
    const glm::vec3 &ray_vector,
    const Face& face,
    const glm::vec3 *vertices
) {
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    const glm::vec3 &a = vertices[face.v1];
    const glm::vec3 &b = vertices[face.v2];
    const glm::vec3 &c = vertices[face.v3];

    glm::vec3 edge1 = b - a;
    glm::vec3 edge2 = c - a;
    glm::vec3 ray_cross_e2 = glm::cross(ray_vector, edge2);
    float det = glm::dot(edge1, ray_cross_e2);
    if (det > -epsilon && det < epsilon) {
        return { false, 0 }; // This ray is parallel to this triangle.
    }        

    float inv_det = 1.0 / det;
    glm::vec3 s = ray_origin - a;
    float u = inv_det * glm::dot(s, ray_cross_e2);
    if ((u < 0 && std::fabs(u) > epsilon) || (u > 1 && std::fabs(u-1) > epsilon)) {
        return { false, 0 };
    }

    glm::vec3 s_cross_e1 = glm::cross(s, edge1);
    float v = inv_det * glm::dot(ray_vector, s_cross_e1);
    if ((v < 0 && std::fabs(v) > epsilon) || (u + v > 1 && std::fabs(u + v - 1) > epsilon)) {
        return { false, 0 };
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = inv_det * glm::dot(edge2, s_cross_e1);
    if (t > epsilon) { // ray intersection
        return { true, t };
    }

    // This means that there is a line intersection but not a ray intersection.
    return { false, 0 };
}

std::tuple<bool, float, float> // mask, t_enter, t_exit
ray_box_intersection(
    const glm::vec3 &ray_origin,
    const glm::vec3 &ray_vector,
    const glm::vec3 &aabb_min,
    const glm::vec3 &aabb_max
) {
    glm::vec3 t1 = (aabb_min - ray_origin) / ray_vector;
    glm::vec3 t2 = (aabb_max - ray_origin) / ray_vector;

    glm::vec3 tmin = glm::min(t1, t2);
    glm::vec3 tmax = glm::max(t1, t2);

    float t_enter = glm::max(tmin.x, glm::max(tmin.y, tmin.z));
    float t_exit = glm::min(tmax.x, glm::min(tmax.y, tmax.z));

    if (t_exit < 0 || t_enter > t_exit) {
        return {false, 0, 0};
    }

    return {true, t_enter, t_exit};
}
