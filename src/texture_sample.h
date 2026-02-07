#pragma once

#include <glm/glm.hpp>
#include "material.h"
#include "cuda_compat.h"

// CPU texture sampling with bilinear interpolation
glm::vec3 sampleTextureCPU(
    const Texture& texture,
    float u, float v
);

// sRGB to linear color space conversion
CUDA_HOST_DEVICE inline glm::vec3 srgbToLinear(const glm::vec3& c) {
    auto toLinear = [](float v) {
        if (v <= 0.04045f) {
            return v / 12.92f;
        }
        return powf((v + 0.055f) / 1.055f, 2.4f);
    };
    return glm::vec3(toLinear(c.x), toLinear(c.y), toLinear(c.z));
}

// GPU texture view structure
struct TextureDeviceView {
    const uint8_t* pixels;
    int32_t width;
    int32_t height;
    int32_t channels;
};

// GPU material view structure
struct MaterialDeviceView {
    glm::vec3 base_color;
    int32_t texture_id;
};

// GPU texture sampling (device function)
CUDA_DEVICE glm::vec3 sampleTextureGPU(
    const TextureDeviceView* textures,
    const MaterialDeviceView* materials,
    int32_t material_idx,
    float u, float v
);
