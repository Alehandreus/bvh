#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "utils.h"

struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<Face> faces;

    // https://learnopengl.com/Model-Loading/Model
    Mesh(const char *scene_path) {
        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(scene_path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            cout << "Assimp error: " << importer.GetErrorString() << endl;
            exit(1);
        }

        vertices.clear();
        faces.clear();

        // cout << scene->mNumMeshes << " meshes; " << scene->mRootNode->mNumChildren << " children" << endl;

        for (int mesh_i = 0; mesh_i < scene->mNumMeshes; mesh_i++) {
            aiMesh *ai_mesh = scene->mMeshes[mesh_i];

            for (int vertex_i = 0; vertex_i < ai_mesh->mNumVertices; vertex_i++) {
                aiVector3D vertex = ai_mesh->mVertices[vertex_i];
                vertices.push_back(glm::vec3(vertex.x, vertex.y, vertex.z));
            }

            for (int face_i = 0; face_i < ai_mesh->mNumFaces; face_i++) {
                aiFace face = ai_mesh->mFaces[face_i];
                faces.push_back({face.mIndices[0], face.mIndices[1], face.mIndices[2]});
            }
        }

        normalize_sphere();
    }

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

        for (int face_i = 0; face_i < faces.size();) {
            Face face = faces[face_i];
            float extent = face.extent(vertices.data());

            if (extent > threshold) {
                glm::vec3 mid1 = (vertices[face[0]] + vertices[face[1]]) / 2.0f;
                glm::vec3 mid2 = (vertices[face[1]] + vertices[face[2]]) / 2.0f;
                glm::vec3 mid3 = (vertices[face[2]] + vertices[face[0]]) / 2.0f;

                Face new_face1 = {face[0], size(vertices), size(vertices) + 2};
                Face new_face2 = {size(vertices), face[1], size(vertices) + 1};
                Face new_face3 = {size(vertices) + 2, size(vertices) + 1, face[2]};
                Face new_face4 = {size(vertices), size(vertices) + 1, size(vertices) + 2};

                faces[face_i] = new_face1;
                faces.push_back(new_face2);
                faces.push_back(new_face3);
                faces.push_back(new_face4);

                vertices.push_back(mid1);
                vertices.push_back(mid2);
                vertices.push_back(mid3);
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
};