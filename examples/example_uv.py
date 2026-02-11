import numpy as np
import torch
from PIL import Image

from mesh_utils import Mesh, GPURayTracer, generate_camera_rays


# ==== Load mesh with UVs and BVH ==== #

mesh = Mesh.from_file("suzanne.fbx", build_bvh=True, max_leaf_size=25)
print(f"Vertices: {mesh.get_num_vertices()}, Faces: {mesh.get_num_faces()}, Has UVs: {mesh.has_uvs()}")

if not mesh.has_uvs():
    print("Warning: mesh has no UV coordinates. The output will be black.")

img_size = 800

# ==== Generate rays ==== #

d_cam_poses, d_dirs = generate_camera_rays(mesh, img_size)


# ==== Ray trace ==== #

ray_tracer = GPURayTracer(mesh)
result = ray_tracer.trace(d_cam_poses, d_dirs)
mask = result.mask
t = result.distance
normals = result.normals
uvs = result.uv

mask = mask.cpu().numpy()
normals = normals.cpu().numpy()
uvs = uvs.cpu().numpy()


# ==== Checkerboard shading using UVs ==== #

freq = 8.0
checker = (np.floor(uvs[:, 0] * freq) + np.floor(uvs[:, 1] * freq)) % 2  # 0 or 1

# Combine checkerboard with simple lighting for depth
light_dir = np.array([1, -1, 1], dtype=np.float32)
light_dir = light_dir / np.linalg.norm(light_dir)

normals[np.isnan(normals)] = 0
lighting = np.dot(normals, light_dir) * 0.5 + 0.5

# Checkerboard colors: dark blue and light yellow
color0 = np.array([0.1, 0.1, 0.4])  # dark
color1 = np.array([1.0, 0.9, 0.5])  # light

colors = np.where(checker[:, None] > 0.5, color1[None, :], color0[None, :])
colors = colors * lighting[:, None]
colors[~mask] = 0

img = colors.reshape(img_size, img_size, 3)
img = np.clip(img * 255, 0, 255).astype(np.uint8)

image = Image.fromarray(img)
image.save('output.png')
print("Saved output.png")
