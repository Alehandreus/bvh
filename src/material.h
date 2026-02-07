#pragma once

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include "cuda_compat.h"

struct Material {
    glm::vec3 base_color;  // RGB color, default (1.0, 1.0, 1.0)
    int32_t texture_id;    // Index into textures array, -1 if none

    Material() : base_color(1.0f, 1.0f, 1.0f), texture_id(-1) {}

    Material(const glm::vec3& color, int32_t tex_id = -1)
        : base_color(color), texture_id(tex_id) {}
};

struct Texture {
    std::vector<uint8_t> pixels;  // RGBA format, 8-bit per channel
    int32_t width;
    int32_t height;
    int32_t channels;             // Always 4 (RGBA)

    Texture() : width(0), height(0), channels(4) {}

    Texture(int32_t w, int32_t h) : width(w), height(h), channels(4) {
        pixels.resize(w * h * 4);
    }
};
