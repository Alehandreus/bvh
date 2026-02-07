#include "texture_sample.h"
#include <cmath>
#include <algorithm>

glm::vec3 sampleTextureCPU(const Texture& texture, float u, float v) {
    // Handle empty texture
    if (texture.pixels.empty() || texture.width == 0 || texture.height == 0) {
        return glm::vec3(1.0f);  // Default white
    }

    // Wrap UV coordinates to [0, 1)
    u = u - floorf(u);
    v = v - floorf(v);

    // Flip V axis (GLTF/OpenGL convention)
    v = 1.0f - v;

    // Calculate pixel coordinates
    float x = u * (texture.width - 1);
    float y = v * (texture.height - 1);

    // Get integer coordinates and interpolation factors
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = std::min(x0 + 1, texture.width - 1);
    int y1 = std::min(y0 + 1, texture.height - 1);

    float tx = x - x0;
    float ty = y - y0;

    // Fetch pixel helper
    auto fetchPixel = [&](int xi, int yi) -> glm::vec3 {
        size_t idx = (yi * texture.width + xi) * 4;
        return glm::vec3(
            texture.pixels[idx + 0] / 255.0f,  // R
            texture.pixels[idx + 1] / 255.0f,  // G
            texture.pixels[idx + 2] / 255.0f   // B
        );
    };

    // Fetch 4 corner pixels
    glm::vec3 c00 = fetchPixel(x0, y0);
    glm::vec3 c10 = fetchPixel(x1, y0);
    glm::vec3 c01 = fetchPixel(x0, y1);
    glm::vec3 c11 = fetchPixel(x1, y1);

    // Bilinear interpolation
    glm::vec3 c0 = glm::mix(c00, c10, tx);
    glm::vec3 c1 = glm::mix(c01, c11, tx);
    glm::vec3 result = glm::mix(c0, c1, ty);

    // Convert from sRGB to linear
    return srgbToLinear(result);
}
