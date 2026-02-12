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

CUDA_HOST_DEVICE inline glm::vec3 vcross(const glm::vec3 &a, const glm::vec3 &b) {
    return glm::vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}


CUDA_HOST_DEVICE inline float vlen2(const glm::vec3 &v) {
    return vdot(v, v);
}


CUDA_HOST_DEVICE inline float vlen(const glm::vec3 &v) {
    return sqrtf(vdot(v, v));
}


CUDA_HOST_DEVICE inline glm::vec3 vmax(const glm::vec3 &a, const glm::vec3 &b) {
    return glm::vec3{
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    };
}



CUDA_HOST_DEVICE inline float dot2(const glm::vec3& v) {
    return vdot(v, v);
}


CUDA_HOST_DEVICE inline glm::vec3 vzero() {
    return glm::vec3{0.0f, 0.0f, 0.0f};
}


CUDA_HOST_DEVICE inline glm::vec3 vclamp(const glm::vec3 &v, float mn, float mx) {
    return glm::vec3{
        fmaxf(mn, fminf(mx, v.x)),
        fmaxf(mn, fminf(mx, v.y)),
        fmaxf(mn, fminf(mx, v.z))
    };
}


CUDA_HOST_DEVICE inline float clamp(float x, float mn, float mx) {
    return fmaxf(mn, fminf(mx, x));
}


// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
CUDA_HOST_DEVICE HitResult ray_triangle_intersection(
    const Ray &ray,
    const Face &face,
    const glm::vec3 *vertices,
    bool allow_negative
) {
    const float epsilon = 1e-8f;

    const glm::vec3 &a = vertices[face.v1];
    const glm::vec3 &b = vertices[face.v2];
    const glm::vec3 &c = vertices[face.v3];

    glm::vec3 edge1 = vsub(b, a);
    glm::vec3 edge2 = vsub(c, a);

    glm::vec3 pvec = vcross(ray.vector, edge2);
    float det = vdot(edge1, pvec);
    if (det > -epsilon && det < epsilon) {
        return {false, 0.0f};
    }

    float inv_det = 1.0f / det;
    glm::vec3 tvec = vsub(ray.origin, a);
    float u = vdot(tvec, pvec) * inv_det;
    if (u < -epsilon || u > 1.0f + epsilon) {
        return {false, 0.0f};
    }

    glm::vec3 qvec = vcross(tvec, edge1);
    float v = vdot(ray.vector, qvec) * inv_det;
    if (v < -epsilon || (u + v) > 1.0f + epsilon) {
        return {false, 0.0f};
    }

    float t = vdot(edge2, qvec) * inv_det;
    if (!allow_negative && t < epsilon) {
        return {false, 0.0f};
    }

    return HitResult(true, t, u, v);
}


CUDA_HOST_DEVICE glm::vec3 ray_triangle_norm(
    const Face &face,
    const glm::vec3 *vertices
) {
    const glm::vec3 &a = vertices[face.v1];
    const glm::vec3 &b = vertices[face.v2];
    const glm::vec3 &c = vertices[face.v3];

    glm::vec3 e1 = vsub(b, a);
    glm::vec3 e2 = vsub(c, a);
    glm::vec3 n  = vcross(e1, e2);

    float len2 = vlen2(n);
    const float eps = 1e-12f;

    if (len2 < eps) {
        return vzero();
    }

    float invLen = 1.0f / sqrtf(len2);
    return vscale(n, invLen);
}


CUDA_HOST_DEVICE HitResult ray_box_intersection(
    const Ray &ray,
    const BBox &bbox,
    bool allow_negative
) {
    const float eps = 1e-8f;

    float tmin = -FLT_MAX;
    float tmax =  FLT_MAX;

    auto update_axis = [&](float o, float d, float mn, float mx) -> bool {
        if (fabsf(d) < eps) {
            // Ray is parallel to this pair of planes
            if (o < mn || o > mx) return false; // no intersection
            return true; // inside slab: no constraint on t from this axis
        }
        float invd = 1.0f / d;
        float t1 = (mn - o) * invd;
        float t2 = (mx - o) * invd;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }

        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;

        return tmin <= tmax;
    };

    if (!update_axis(ray.origin.x, ray.vector.x, bbox.min.x, bbox.max.x)) return {false, 0.0f};
    if (!update_axis(ray.origin.y, ray.vector.y, bbox.min.y, bbox.max.y)) return {false, 0.0f};
    if (!update_axis(ray.origin.z, ray.vector.z, bbox.min.z, bbox.max.z)) return {false, 0.0f};

    // Now tmin = t_enter, tmax = t_exit
    float t_enter = tmin;
    float t_exit  = tmax;

    if (!allow_negative) {
        if (t_exit < 0.0f) return {false, 0.0f};     // box entirely behind ray
        if (t_enter < 0.0f) t_enter = 0.0f;          // ray starts inside box
    }

    return {true, t_enter};
}


// https://iquilezles.org/articles/distfunctions/
CUDA_HOST_DEVICE float box_df(
    const glm::vec3& p,
    const BBox& box
) {
    glm::vec3 d = vmax(
        vmax(vsub(box.min, p), vsub(p, box.max)),
        vzero()
    );
    return vlen(d);
}


CUDA_HOST_DEVICE inline glm::vec3 closest_point_on_segment(
    const glm::vec3& p,
    const glm::vec3& a,
    const glm::vec3& b
) {
    glm::vec3 ab = vsub(b, a);
    float denom = dot2(ab);

    float t = 0.0f;
    if (denom > 0.0f) {
        t = clamp(vdot(vsub(p, a), ab) / denom, 0.0f, 1.0f);
    }

    return vadd(a, vscale(ab, t));
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

    const glm::vec3 ab = vsub(b, a);
    const glm::vec3 ac = vsub(c, a);
    const glm::vec3 ap = vsub(p, a);

    // // Handle degenerate triangles (zero area) by falling back to the closest edge/vertex
    const glm::vec3 n  = vcross(ab, ac);
    const float n2 = dot2(n);
    if (n2 <= 0.0f) {
        glm::vec3 q_ab = closest_point_on_segment(p, a, b);
        glm::vec3 q_bc = closest_point_on_segment(p, b, c);
        glm::vec3 q_ca = closest_point_on_segment(p, c, a);

        float d2_ab = dot2(vsub(p, q_ab));
        float d2_bc = dot2(vsub(p, q_bc));
        float d2_ca = dot2(vsub(p, q_ca));

        glm::vec3 q = q_ab;
        float d2 = d2_ab;

        if (d2_bc < d2) { d2 = d2_bc; q = q_bc; }
        if (d2_ca < d2) { d2 = d2_ca; q = q_ca; }

        SDFHitResult out;
        out.t = sqrtf(d2);
        out.closest = q;

        // barycentrics (best-effort; may be unstable if extremely degenerate)
        glm::vec3 v0 = vsub(b, a);
        glm::vec3 v1 = vsub(c, a);
        glm::vec3 v2 = vsub(q, a);

        float d00 = vdot(v0, v0);
        float d01 = vdot(v0, v1);
        float d11 = vdot(v1, v1);
        float d20 = vdot(v2, v0);
        float d21 = vdot(v2, v1);

        float denom = d00 * d11 - d01 * d01;
        if (denom != 0.0f) {
            float inv = 1.0f / denom;
            float v = (d11 * d20 - d01 * d21) * inv;
            float w = (d00 * d21 - d01 * d20) * inv;
            float u = 1.0f - v - w;
            out.barycentrics = vmake(u, v, w);
        } else {
            out.barycentrics = vmake(1.0f, 0.0f, 0.0f);
        }

        return out;
    }

    // ---- Ericson region tests ----
    float d1 = vdot(ab, ap);
    float d2 = vdot(ac, ap);

    glm::vec3 bp = vsub(p, b);
    float d3 = vdot(ab, bp);
    float d4 = vdot(ac, bp);

    glm::vec3 cp = vsub(p, c);
    float d5 = vdot(ab, cp);
    float d6 = vdot(ac, cp);

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
                q = vadd(a, vscale(ab, v));
            } else {
                // Vertex region C
                if (d6 >= 0.0f && d5 <= d6) {
                    q = c;
                } else {
                    // Edge AC
                    float vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                        float w = d2 / (d2 - d6);
                        q = vadd(a, vscale(ac, w));
                    } else {
                        // Edge BC
                        float va = d3 * d6 - d5 * d4;
                        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                            glm::vec3 bc = vsub(c, b);
                            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            q = vadd(b, vscale(bc, w));
                        } else {
                            // Inside face region – project to plane
                            float invDen = 1.0f / (va + vb + vc);
                            float v = vb * invDen;
                            float w = vc * invDen;
                            q = vadd(a, vadd(vscale(ab, v), vscale(ac, w)));
                        }
                    }
                }
            }
        }
    }

    // Signed distance: sign by triangle plane orientation (two-sided surface)
    glm::vec3 pq = vsub(p, q);
    float sd = vlen(pq);
    float side = vdot(n, pq); // no need to normalize n
    if (side < 0.0f) sd = -sd;

    SDFHitResult out;
    out.t = sd;
    out.closest = q;

    // barycentrics
    glm::vec3 v0 = vsub(b, a);
    glm::vec3 v1 = vsub(c, a);
    glm::vec3 v2 = vsub(q, a);

    float d00 = vdot(v0, v0);
    float d01 = vdot(v0, v1);
    float d11 = vdot(v1, v1);
    float d20 = vdot(v2, v0);
    float d21 = vdot(v2, v1);

    float denom = d00 * d11 - d01 * d01;
    if (denom != 0.0f) {
        float inv = 1.0f / denom;
        float v = (d11 * d20 - d01 * d21) * inv;
        float w = (d00 * d21 - d01 * d20) * inv;
        float u = 1.0f - v - w;
        out.barycentrics = vmake(u, v, w);
    } else {
        out.barycentrics = vmake(1.0f, 0.0f, 0.0f);
    }

    return out;
}
