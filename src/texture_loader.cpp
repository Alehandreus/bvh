#include "texture_loader.h"

#include <assimp/scene.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

bool loadImageFromFile(const std::string& path, Texture* outTexture, std::string* error) {
    int width, height, channels;

    // Load image using stb_image
    // Force 4 channels (RGBA)
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 4);

    if (!data) {
        if (error) {
            *error = "Failed to load texture from file: " + path + " - " + stbi_failure_reason();
        }
        return false;
    }

    // Populate texture struct
    outTexture->width = width;
    outTexture->height = height;
    outTexture->channels = 4;  // Always RGBA
    outTexture->pixels.resize(width * height * 4);
    std::memcpy(outTexture->pixels.data(), data, width * height * 4);

    // Free stbi data
    stbi_image_free(data);

    return true;
}

bool loadImageFromEmbedded(const aiTexture* texture, Texture* outTexture, std::string* error) {
    if (!texture) {
        if (error) {
            *error = "Null texture pointer";
        }
        return false;
    }

    // Check if texture is compressed (height == 0) or uncompressed
    if (texture->mHeight == 0) {
        // Compressed texture (PNG, JPG, etc.)
        int width, height, channels;

        // Load from memory buffer, force 4 channels
        unsigned char* data = stbi_load_from_memory(
            reinterpret_cast<const unsigned char*>(texture->pcData),
            texture->mWidth,  // mWidth contains data size for compressed textures
            &width, &height, &channels, 4
        );

        if (!data) {
            if (error) {
                *error = "Failed to decode embedded texture - " + std::string(stbi_failure_reason());
            }
            return false;
        }

        // Populate texture struct
        outTexture->width = width;
        outTexture->height = height;
        outTexture->channels = 4;
        outTexture->pixels.resize(width * height * 4);
        std::memcpy(outTexture->pixels.data(), data, width * height * 4);

        // Free stbi data
        stbi_image_free(data);

    } else {
        // Uncompressed texture (raw RGBA data)
        outTexture->width = texture->mWidth;
        outTexture->height = texture->mHeight;
        outTexture->channels = 4;

        size_t num_pixels = texture->mWidth * texture->mHeight;
        outTexture->pixels.resize(num_pixels * 4);

        // Convert aiTexel (RGBA, 8-bit) to our format
        for (size_t i = 0; i < num_pixels; i++) {
            const aiTexel& texel = texture->pcData[i];
            outTexture->pixels[i * 4 + 0] = texel.r;
            outTexture->pixels[i * 4 + 1] = texel.g;
            outTexture->pixels[i * 4 + 2] = texel.b;
            outTexture->pixels[i * 4 + 3] = texel.a;
        }
    }

    return true;
}
