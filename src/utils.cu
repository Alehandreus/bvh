#include "utils.h"

#include <fstream>
#include <iostream>

using std::cin, std::cout, std::endl;

#include <iostream>
#include <vector>
#include <chrono>

#include <glm/glm.hpp>

#include "cuda_compat.h"

using std::cout, std::endl;

CUDA_HOST_DEVICE Ray::Ray(const glm::vec3 &origin, const glm::vec3 &vector) : origin(origin), vector(vector) {}

// size_t to uint32_t causes narrowing conversion warning
// no templates this time
uint32_t size(const std::vector<float> &v) {
    return v.size();
}
uint32_t size(const std::vector<uint32_t> &v) {
    return v.size();
}
uint32_t size(const std::vector<glm::vec3> &v) {
    return v.size();
}

Face::Face(uint32_t v1, uint32_t v2, uint32_t v3) : v1(v1), v2(v2), v3(v3) {}

void Face::calc_centroid(const glm::vec3 *vertices) {
    centroid = (vertices[v1] + vertices[v2] + vertices[v3]) / 3.0f;
}

float Face::extent(const glm::vec3 *vertices) const {
    glm::vec3 min = vertices[v1];
    glm::vec3 max = vertices[v1];

    for (int i = 1; i < 3; i++) {
        min = glm::min(min, vertices[operator[](i)]);
        max = glm::max(max, vertices[operator[](i)]);
    }

    return glm::length(max - min);
}

uint32_t Face::operator[](uint32_t i) const {
    switch (i) {
        case 0: return v1;
        case 1: return v2;
        case 2: return v3;
        default: return 0;
    }
}

CUDA_HOST_DEVICE BBox::BBox() : min(FLT_MAX), max(-FLT_MAX) {}

CUDA_HOST_DEVICE void BBox::update(const glm::vec3 &point) {
    min = glm::min(min, point);
    max = glm::max(max, point);
}

CUDA_HOST_DEVICE bool BBox::inside(const glm::vec3 &point) const {
    return point.x >= min.x && point.x <= max.x &&
            point.y >= min.y && point.y <= max.y &&
            point.z >= min.z && point.z <= max.z;
}

CUDA_HOST_DEVICE glm::vec3 BBox::diagonal() const {
    return max - min;
}

CUDA_HOST_DEVICE HitResult::HitResult() : hit(false), t(0) {}

CUDA_HOST_DEVICE HitResult::HitResult(bool hit, float t) : hit(hit), t(t) {}

CUDA_HOST_DEVICE HitResult::HitResult(bool hit, float t1, float t2) : hit(hit), t1(t1), t2(t2) {}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
CUDA_HOST_DEVICE HitResult ray_triangle_intersection(
    const Ray &ray,
    const Face& face,
    const glm::vec3 *vertices
) {
    float epsilon = 0.0000001;

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
    if (t > epsilon) { // ray intersection
        return { true, t };
    }

    // This means that there is a line intersection but not a ray intersection.
    return { false, 0 };
}

CUDA_HOST_DEVICE HitResult ray_box_intersection(
    const Ray &ray,
    const BBox &bbox
) {
    glm::vec3 t1 = (bbox.min - ray.origin) / ray.vector;
    glm::vec3 t2 = (bbox.max - ray.origin) / ray.vector;

    glm::vec3 tmin = glm::min(t1, t2);
    glm::vec3 tmax = glm::max(t1, t2);

    float t_enter = glm::max(tmin.x, glm::max(tmin.y, tmin.z));
    float t_exit = glm::min(tmax.x, glm::min(tmax.y, tmax.z));

    if (t_exit < 0 || t_enter > t_exit) {
        return {false, 0, 0};
    }

    return {true, t_enter, t_exit};
}

// thanks copilot
void save_to_bmp(const unsigned int *pixels, int width, int height, const char* filename) {
    // File header (14 bytes)
    unsigned char fileHeader[14] = {
        'B', 'M', // Magic identifier
        0, 0, 0, 0, // File size in bytes, will be set later
        0, 0, // Reserved
        0, 0, // Reserved
        54, 0, 0, 0 // Offset to image data, 54 bytes
    };

    // Information header (40 bytes)
    unsigned char infoHeader[40] = {
        40, 0, 0, 0, // Header size
        0, 0, 0, 0, // Image width, will be set later
        0, 0, 0, 0, // Image height, will be set later
        1, 0, // Number of color planes
        24, 0, // Bits per pixel
        0, 0, 0, 0, // Compression type (0 = none)
        0, 0, 0, 0, // Image size (can be 0 for no compression)
        0, 0, 0, 0, // X pixels per meter (not specified)
        0, 0, 0, 0, // Y pixels per meter (not specified)
        0, 0, 0, 0, // Number of colors (0 = default)
        0, 0, 0, 0  // Important colors (0 = all)
    };

    // Set width and height in infoHeader
    *(int*)&infoHeader[4] = width;
    *(int*)&infoHeader[8] = height;

    // The row length in bytes, each pixel needs 3 bytes, rows are padded to be a multiple of 4 bytes
    int rowPadding = (4 - (width * 3 % 4)) % 4;
    int rowSize = width * 3 + rowPadding;
    int dataSize = rowSize * height;

    // Set file size in fileHeader
    *(int*)&fileHeader[2] = 54 + dataSize;

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        cout << "Could not open file for writing." << endl;
        return;
    }

    // Write headers
    file.write((char*)fileHeader, 14);
    file.write((char*)infoHeader, 40);

    // Write pixel data
    for (int y = height - 1; y >= 0; y--) { // BMP files store data bottom-up
        for (int x = 0; x < width; x++) {
            unsigned int pixel = pixels[y * width + x];
            unsigned char colors[3] = {
                (unsigned char)((pixel >> 0) & 0xFF), // Blue
                (unsigned char)((pixel >> 8) & 0xFF), // Green
                (unsigned char)((pixel >> 16) & 0xFF) // Red
            };
            file.write((char*)colors, 3);
        }
        if (rowPadding > 0) {
            static const unsigned char padding[3] = { 0, 0, 0 };
            file.write((char*)padding, rowPadding);
        }
    }

    file.close();
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
