#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

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
            return {false, 0};
        }
        ray2.vector.x = FLT_MAX;
    }
    if (fabs(ray.vector.y) < eps) {
        if (ray2.origin.y < bbox.min.y || ray2.origin.y > bbox.max.y) {
            return {false, 0};
        }
        ray2.vector.y = FLT_MAX;
    }
    if (fabs(ray.vector.z) < eps) {
        if (ray2.origin.z < bbox.min.z || ray2.origin.z > bbox.max.z) {
            return {false, 0};
        }
        ray2.vector.z = FLT_MAX;
    }

    glm::vec3 t1 = (bbox.min - ray2.origin) / ray2.vector;
    glm::vec3 t2 = (bbox.max - ray2.origin) / ray2.vector;

    glm::vec3 tmin = glm::min(t1, t2);
    glm::vec3 tmax = glm::max(t1, t2);

    float t_enter = glm::max(tmin.x, glm::max(tmin.y, tmin.z));
    float t_exit = glm::min(tmax.x, glm::min(tmax.y, tmax.z));

    if (t_enter > t_exit || (t_exit < 0 && !allow_negative)) {
        return {false, 0};
    }

    return {true, t_enter};
}

// https://iquilezles.org/articles/distfunctions/
CUDA_HOST_DEVICE float box_df(
    const glm::vec3& p,
    const BBox& box
) {
    glm::vec3 d = glm::max(glm::max(box.min - p, p - box.max), glm::vec3(0.0f));
    return glm::length(d);
}

CUDA_HOST_DEVICE inline float dot2(const glm::vec3& v) {
    return glm::dot(v, v);
}

CUDA_HOST_DEVICE inline glm::vec3 closest_point_on_segment(
    const glm::vec3& p,
    const glm::vec3& a,
    const glm::vec3& b
) {
    glm::vec3 ab = b - a;
    float denom = dot2(ab);
    float t = 0.0f;
    if (denom > 0.0f) {
        t = glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
    }
    return a + t * ab;
}

// Robust closest-point on triangle (Ericson). Also returns signed distance.
CUDA_HOST_DEVICE SDFHitResult triangle_sdf(
    const glm::vec3 &p,
    const Face& face,
    const glm::vec3 *vertices
) {
    const glm::vec3 &a = vertices[face.v1];
    const glm::vec3 &b = vertices[face.v2];
    const glm::vec3 &c = vertices[face.v3];

    const glm::vec3 ab = b - a;
    const glm::vec3 ac = c - a;
    const glm::vec3 ap = p - a;

    // Handle degenerate triangles (zero area) by falling back to the closest edge/vertex
    const glm::vec3 n  = glm::cross(ab, ac);
    const float n2 = dot2(n);
    if (n2 <= 0.0f) {
        // collapse to best of the three segments
        glm::vec3 q_ab = closest_point_on_segment(p, a, b);
        glm::vec3 q_bc = closest_point_on_segment(p, b, c);
        glm::vec3 q_ca = closest_point_on_segment(p, c, a);

        float d2_ab = dot2(p - q_ab);
        float d2_bc = dot2(p - q_bc);
        float d2_ca = dot2(p - q_ca);

        glm::vec3 q = q_ab;
        float d2 = d2_ab;

        if (d2_bc < d2) { d2 = d2_bc; q = q_bc; }
        if (d2_ca < d2) { d2 = d2_ca; q = q_ca; }

        // No meaningful normal => return unsigned distance; sign = +1
        SDFHitResult out;
        out.t = glm::sqrt(d2);
        out.closest = q;
        return out;
    }

    // ---- Ericson region tests ----
    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);

    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);

    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);

    glm::vec3 q; // closest point

    // Vertex region A
    if (d1 <= 0.0f && d2 <= 0.0f) {
        q = a;
    } else {
        // Vertex region B
        if (d3 >= 0.0f && d4 <= d3) {
            q = b;
        } else {
            // Edge AB
            float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                float v = d1 / (d1 - d3);
                q = a + v * ab;
            } else {
                // Vertex region C
                if (d6 >= 0.0f && d5 <= d6) {
                    q = c;
                } else {
                    // Edge AC
                    float vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                        float w = d2 / (d2 - d6);
                        q = a + w * ac;
                    } else {
                        // Edge BC
                        float va = d3 * d6 - d5 * d4;
                        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                            glm::vec3 bc = c - b;
                            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            q = b + w * bc;
                        } else {
                            // Inside face region – project to plane
                            float invDen = 1.0f / (va + vb + vc);
                            float v = vb * invDen;
                            float w = vc * invDen;
                            q = a + v * ab + w * ac;
                        }
                    }
                }
            }
        }
    }

    // Signed distance: sign by triangle plane orientation (two-sided surface)
    // Negative if p is on the opposite side from the normal (ab x ac).
    float sd = glm::length(p - q);
    float side = glm::dot(n, p - q);     // no need to normalize n
    if (side < 0.0f) sd = -sd;

    SDFHitResult out;
    out.t = sd;
    out.closest = q;
    return out;
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
