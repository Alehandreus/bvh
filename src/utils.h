#include <glm/glm.hpp>

#include <vector>
#include <tuple>

struct Ray {
    glm::vec3 origin;
    glm::vec3 vector;

    Ray(const glm::vec3 &origin, const glm::vec3 &vector) : origin(origin), vector(vector) {}
};

struct Face {
    uint32_t v1, v2, v3;
    glm::vec3 centroid;

    Face(uint32_t v1, uint32_t v2, uint32_t v3) : v1(v1), v2(v2), v3(v3) {}

    void calc_centroid(const glm::vec3 *vertices) {
        centroid = (vertices[v1] + vertices[v2] + vertices[v3]) / 3.0f;
    }

    float extent(const glm::vec3 *vertices) const {
        glm::vec3 min = vertices[v1];
        glm::vec3 max = vertices[v1];

        for (int i = 1; i < 3; i++) {
            min = glm::min(min, vertices[operator[](i)]);
            max = glm::max(max, vertices[operator[](i)]);
        }

        return glm::length(max - min);
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

struct BBox {
    glm::vec3 min, max;

    BBox() : min(FLT_MAX), max(-FLT_MAX) {}

    void update(const glm::vec3 &point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    bool inside(const glm::vec3 &point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }

    glm::vec3 diagonal() const {
        return max - min;
    }
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

    HitResult() : hit(false), t(0) {}
    HitResult(bool hit, float t) : hit(hit), t(t) {}
    HitResult(bool hit, float t1, float t2) : hit(hit), t1(t1), t2(t2) {}
};

HitResult ray_triangle_intersection(
    const Ray &ray,
    const Face& face,
    const glm::vec3 *vertices
);

HitResult ray_box_intersection(
    const Ray &ray,
    const BBox &bbox
);

void SaveToBMP(const unsigned int *pixels, int width, int height, const char* filename);
