#include <glm/glm.hpp>

#include <vector>
#include <tuple>

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

std::tuple<bool, float> // mask, t
ray_triangle_intersection(
    const glm::vec3 &ray_origin,
    const glm::vec3 &ray_vector,
    const Face& face,
    const glm::vec3 *vertices
);

std::tuple<bool, float, float> // mask, t_enter, t_exit
ray_box_intersection(
    const glm::vec3 &ray_origin,
    const glm::vec3 &ray_vector,
    const glm::vec3 &aabb_min,
    const glm::vec3 &aabb_max
);

void SaveToBMP(const unsigned int *pixels, int width, int height, const char* filename);