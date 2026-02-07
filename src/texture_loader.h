#pragma once

#include <string>
#include "material.h"

// Forward declaration for Assimp texture type
struct aiTexture;

// Load texture from external file (PNG, JPG, etc.)
// Returns true on success, false on failure
// On failure, error message is written to error parameter
bool loadImageFromFile(const std::string& path, Texture* outTexture, std::string* error);

// Load texture from embedded GLTF data
// Returns true on success, false on failure
// On failure, error message is written to error parameter
bool loadImageFromEmbedded(const aiTexture* texture, Texture* outTexture, std::string* error);
