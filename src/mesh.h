#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <fstream>
#include <limits>

#include <glm/gtc/matrix_transform.hpp>

#include "utils.h"

#include "stb_image_write.h"
#include "material.h"
#include "texture_loader.h"

#include <unordered_map>
#include <filesystem>

struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<Face> faces;
    std::vector<Material> materials;
    std::vector<Texture> textures;

    Mesh(const char *scene_path, bool swap_yz = false) {
        Assimp::Importer importer;
        unsigned int flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_PreTransformVertices;
        const aiScene *scene = importer.ReadFile(scene_path, flags);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            cout << "Assimp error: " << importer.GetErrorString() << endl;
            exit(1);
        }

        vertices.clear();
        faces.clear();

        // Load materials and textures
        std::unordered_map<std::string, int> textureCache;  // filename -> texture ID
        std::filesystem::path baseDir = std::filesystem::path(scene_path).parent_path();

        // Helper to load texture
        auto loadTexture = [&](const aiString& texPath, bool srgb) -> int32_t {
            std::string key = texPath.C_Str();

            // Check cache first
            auto it = textureCache.find(key);
            if (it != textureCache.end()) return it->second;

            // Load new texture
            Texture texture;
            std::string error;
            bool loaded = false;

            if (!key.empty() && key[0] == '*') {
                // Embedded texture (format: "*0", "*1", etc.)
                int index = std::atoi(key.c_str() + 1);
                if (index >= 0 && index < (int)scene->mNumTextures) {
                    loaded = loadImageFromEmbedded(scene->mTextures[index], &texture, &error);
                }
            } else {
                // File texture (relative to model directory)
                std::filesystem::path texFile = baseDir / key;
                loaded = loadImageFromFile(texFile.string(), &texture, &error);
            }

            if (!loaded) {
                cout << "Warning: Failed to load texture: " << error << endl;
                return -1;
            }

            // Cache and return ID
            int32_t id = (int32_t)textures.size();
            cout << "Loaded texture " << id << ": " << texture.width << "x" << texture.height
                 << " (" << texture.pixels.size() << " bytes)" << endl;
            textures.push_back(std::move(texture));
            textureCache[key] = id;
            return id;
        };

        // Load materials
        cout << "Loading " << scene->mNumMaterials << " materials, "
             << scene->mNumTextures << " embedded textures" << endl;
        materials.resize(scene->mNumMaterials);
        for (unsigned int mat_i = 0; mat_i < scene->mNumMaterials; mat_i++) {
            aiMaterial* mat = scene->mMaterials[mat_i];

            // Get base color
            aiColor3D base_color(1.0f, 1.0f, 0.3f);
            mat->Get(AI_MATKEY_COLOR_DIFFUSE, base_color);
            if (base_color.IsBlack()) {
                mat->Get(AI_MATKEY_BASE_COLOR, base_color);
            }
            materials[mat_i].base_color = glm::vec3(base_color.r, base_color.g, base_color.b);

            // Get texture
            aiString texPath;
            materials[mat_i].texture_id = -1;
            if (mat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS ||
                mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
                cout << "Material " << mat_i << " texture path: " << texPath.C_Str() << endl;
                materials[mat_i].texture_id = loadTexture(texPath, true);  // sRGB
                cout << "Material " << mat_i << " assigned texture_id=" << materials[mat_i].texture_id << endl;
            } else {
                cout << "Material " << mat_i << " has no texture" << endl;
            }
        }

        // Load meshes
        for (int mesh_i = 0; mesh_i < scene->mNumMeshes; mesh_i++) {
            aiMesh *ai_mesh = scene->mMeshes[mesh_i];
            uint32_t base_index = static_cast<uint32_t>(vertices.size());
            if (vertices.size() > std::numeric_limits<uint32_t>::max() ||
                ai_mesh->mNumVertices > std::numeric_limits<uint32_t>::max() - base_index) {
                std::cerr << "Mesh has too many vertices for uint32 indices." << std::endl;
                exit(1);
            }

            bool has_uvs = ai_mesh->mTextureCoords[0] != nullptr;
            int32_t material_idx = (ai_mesh->mMaterialIndex < materials.size()) ?
                                   (int32_t)ai_mesh->mMaterialIndex : -1;

            for (int vertex_i = 0; vertex_i < ai_mesh->mNumVertices; vertex_i++) {
                aiVector3D vertex = ai_mesh->mVertices[vertex_i];
                if (!swap_yz) {
                    vertices.push_back(glm::vec3(vertex.x, vertex.y, vertex.z));
                } else {
                    vertices.push_back(glm::vec3(vertex.x, -vertex.z, vertex.y));
                }

                if (has_uvs) {
                    aiVector3D uv = ai_mesh->mTextureCoords[0][vertex_i];
                    uvs.push_back(glm::vec2(uv.x, uv.y));
                }
            }

            for (int face_i = 0; face_i < ai_mesh->mNumFaces; face_i++) {
                aiFace face = ai_mesh->mFaces[face_i];
                if (face.mNumIndices < 3) {
                    continue;
                }
                faces.push_back(Face(
                    base_index + face.mIndices[0],
                    base_index + face.mIndices[1],
                    base_index + face.mIndices[2],
                    material_idx
                ));
            }
        }
    }

    Mesh(const std::vector<glm::vec3> vertices, const std::vector<Face> faces, const std::vector<glm::vec2> uvs = {})
        : vertices(std::move(vertices)), faces(std::move(faces)), uvs(std::move(uvs)) {}

    void print_stats() {
        cout << vertices.size() << " vertices; " << faces.size() << " faces" << endl; // why are vertices duplicated ????
    }

    void normalize_sphere() {
        glm::vec3 v_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        glm::vec3 v_min = {FLT_MAX, FLT_MAX, FLT_MAX};

        for (int i = 0; i < vertices.size(); ++i) {
            v_max = glm::max(v_max, vertices[i]);
            v_min = glm::min(v_min, vertices[i]);
        }

        glm::vec3 v_center = (v_max + v_min) * 0.5f;

        for (int i = 0; i < vertices.size(); ++i) {
            vertices[i] = vertices[i] - v_center;
        }

        float max_dist = -FLT_MAX;
        for (int i = 0; i < vertices.size(); ++i) {
            float length = glm::length(vertices[i]);
            max_dist = fmax(max_dist, length);
        }

        float v_scale = 1 / max_dist;

        for (int i = 0; i < vertices.size(); ++i) {
            vertices[i] = vertices[i] * v_scale;
        }
    }

    void split_faces(float frac) {
        std::vector<float> extents;
        for (int i = 0; i < faces.size(); i++) {
            Face &face = faces[i];
            float extent = face.extent(vertices.data());
            extents.push_back(extent);
        }

        std::sort(extents.begin(), extents.end());

        float threshold = extents[extents.size() * frac];

        int initial_faces_size = faces.size();

        for (int face_i = 0; face_i < ((frac == 0) ? initial_faces_size : faces.size());) {
            Face face = faces[face_i];
            float extent = face.extent(vertices.data());

            if (extent > threshold) {
                glm::vec3 mid1 = (vertices[face[0]] + vertices[face[1]]) / 2.0f;
                glm::vec3 mid2 = (vertices[face[1]] + vertices[face[2]]) / 2.0f;
                glm::vec3 mid3 = (vertices[face[2]] + vertices[face[0]]) / 2.0f;

                Face new_face1 = {face[0], size(vertices), size(vertices) + 2, face.material_idx};
                Face new_face2 = {size(vertices), face[1], size(vertices) + 1, face.material_idx};
                Face new_face3 = {size(vertices) + 2, size(vertices) + 1, face[2], face.material_idx};
                Face new_face4 = {size(vertices), size(vertices) + 1, size(vertices) + 2, face.material_idx};

                faces[face_i] = new_face1;
                faces.push_back(new_face2);
                faces.push_back(new_face3);
                faces.push_back(new_face4);

                vertices.push_back(mid1);
                vertices.push_back(mid2);
                vertices.push_back(mid3);

                if (!uvs.empty()) {
                    glm::vec2 uv_mid1 = (uvs[face[0]] + uvs[face[1]]) / 2.0f;
                    glm::vec2 uv_mid2 = (uvs[face[1]] + uvs[face[2]]) / 2.0f;
                    glm::vec2 uv_mid3 = (uvs[face[2]] + uvs[face[0]]) / 2.0f;
                    uvs.push_back(uv_mid1);
                    uvs.push_back(uv_mid2);
                    uvs.push_back(uv_mid3);
                }
            } else {
                face_i++;
            }
        }
    }

    std::tuple<glm::vec3, glm::vec3> bounds() {
        glm::vec3 min(FLT_MAX);
        glm::vec3 max(-FLT_MAX);

        for (const glm::vec3 &vertex : vertices) {
            min = glm::min(min, vertex);
            max = glm::max(max, vertex);
        }

        return {min, max};
    }

    void save_to_obj(const char *filename) {
        std::ofstream outFile(filename);

        if (!outFile.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        for (const glm::vec3 &vertex : vertices) {
            // outFile << "v " << vertex.x << " " << vertex.y << " " << vertex.z << "\n";

            // swap y and z for blender
            outFile << "v " << vertex.x << " " << vertex.z << " " << -vertex.y << "\n";
        }

        for (const Face &face : faces) {
            outFile << "f " << face.v1 + 1 << " " << face.v2 + 1 << " " << face.v3 + 1 << "\n";
        }

        outFile.close();
    }

    glm::vec3 get_c() const {
        glm::vec3 c(0.0f);
        for (auto& v : vertices) c += v;
        c /= float(vertices.size());

        // c = {c.x, c.z, -c.y}; // swap y and z for blender
        c = {c.x, c.y, c.z};

        return c;
    }

    float get_R() const {
        glm::vec3 c = get_c();

        float R = 0.0f;
        // for (auto& v : vertices) R = std::max(R, glm::length(v - glm::vec3(c.x, -c.z, c.y)));
        for (auto& v : vertices) R = std::max(R, glm::length(v - glm::vec3(c.x, c.y, c.z)));
        if (R <= 0.0f) R = 1.0f;

        return R;
    }

    bool save_preview(const char *filename, int width, int height, glm::vec3 c, float R) const {
        if (vertices.empty() || faces.empty() || width <= 0 || height <= 0) return false;

        // Center & scale heuristics
        // glm::vec3 c(0.0f);
        // for (auto& v : vertices) c += v;
        // c /= float(vertices.size());
        // float R = 0.0f;
        // for (auto& v : vertices) R = std::max(R, glm::length(v - c));
        // if (R <= 0.0f) R = 1.0f;

        // c = {c.x, c.z, -c.y};

        // Camera: front-right, slightly above, looking at the mesh center
        glm::vec3 eye = c + glm::normalize(glm::vec3(1.0f, 0.35f, 1.0f)) * (2.2f * R);
        glm::mat4 V   = glm::lookAt(eye, c, glm::vec3(0, 1, 0));
        float aspect  = float(width) / float(height);
        float nearZ   = std::max(1e-4f, 0.10f * R);
        float farZ    = nearZ + 6.0f * R;
        glm::mat4 P   = glm::perspective(glm::radians(50.0f), aspect, nearZ, farZ);
        glm::mat4 MVP = P * V;

        std::vector<float> zbuf(size_t(width) * size_t(height), std::numeric_limits<float>::infinity());
        std::vector<uint8_t> rgb(size_t(width) * size_t(height) * 3u, 0);

        auto to_screen = [&](const glm::vec3& p, float& sx, float& sy, float& z01, float& w) {
            glm::vec4 clip = MVP * glm::vec4(p, 1.0f);
            w = clip.w;
            if (w <= 0.0f) return false; // trivially reject behind near plane
            glm::vec3 ndc = glm::vec3(clip) / w;      // [-1,1]
            sx = (ndc.x * 0.5f + 0.5f) * (width  - 1);
            sy = (1.0f - (ndc.y * 0.5f + 0.5f)) * (height - 1); // flip Y
            z01 = ndc.z * 0.5f + 0.5f;                 // [0,1], smaller = closer
            return ndc.x >= -1.5f && ndc.x <= 1.5f && ndc.y >= -1.5f && ndc.y <= 1.5f;
        };

        auto edge = [](const glm::vec2& a, const glm::vec2& b, const glm::vec2& p) -> float {
            return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
        };

        for (const Face& f : faces) {
            glm::vec3 v0 = vertices[f.v1];
            glm::vec3 v1 = vertices[f.v2];
            glm::vec3 v2 = vertices[f.v3];
            
            // v0 = {v0.x, v0.z, -v0.y}; // swap y and z for blender
            // v1 = {v1.x, v1.z, -v1.y};
            // v2 = {v2.x, v2.z, -v2.y};

            // Face normal/shading in world space (flat shading)
            glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            if (!std::isfinite(n.x)) continue;
            glm::vec3 fc = (v0 + v1 + v2) * (1.0f / 3.0f);
            glm::vec3 L  = glm::normalize(eye - fc);
            // float shade  = std::clamp(glm::dot(n, L), 0.0f, 1.0f);
            float shade = glm::dot(n, L) * 0.5f + 0.5f;

            // Back-face cull (optional, keeps front-facing triangles only)
            if (glm::dot(n, L) <= 0.0f) continue;

            // Project to screen
            float x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2;
            if (!to_screen(v0, x0, y0, z0, w0)) continue;
            if (!to_screen(v1, x1, y1, z1, w1)) continue;
            if (!to_screen(v2, x2, y2, z2, w2)) continue;

            glm::vec2 p0(x0, y0), p1(x1, y1), p2(x2, y2);
            float area = edge(p0, p1, p2);
            if (area == 0.0f) continue;

            int minx = std::max(0, int(std::floor(std::min({x0, x1, x2}))));
            int maxx = std::min(width - 1,  int(std::ceil (std::max({x0, x1, x2}))));
            int miny = std::max(0, int(std::floor(std::min({y0, y1, y2}))));
            int maxy = std::min(height - 1, int(std::ceil (std::max({y0, y1, y2}))));

            for (int y = miny; y <= maxy; ++y) {
                for (int x = minx; x <= maxx; ++x) {
                    glm::vec2 p(x + 0.5f, y + 0.5f);
                    float w0b = edge(p1, p2, p) / area;
                    float w1b = edge(p2, p0, p) / area;
                    float w2b = edge(p0, p1, p) / area;
                    if (w0b < 0.0f || w1b < 0.0f || w2b < 0.0f) continue;

                    // Interpolate depth (already perspective-divided -> screen-space)
                    float z = w0b * z0 + w1b * z1 + w2b * z2;
                    size_t idx = size_t(y) * size_t(width) + size_t(x);
                    if (z < zbuf[idx]) {
                        zbuf[idx] = z;
                        uint8_t g = (uint8_t)std::lround(255.0f * shade);
                        rgb[3 * idx + 0] = g;
                        rgb[3 * idx + 1] = g;
                        rgb[3 * idx + 2] = g;
                    }
                }
            }
        }

        stbi_write_png(filename, width, height, 3, rgb.data(), width * 3);

        return true;
    }    
};
