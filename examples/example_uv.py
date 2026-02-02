import numpy as np
import torch
from PIL import Image

from mesh_utils import Mesh, CPUBuilder, GPUTraverser
from mesh_utils import GPURayTracer


# ==== Load mesh with UVs ==== #

mesh = Mesh.from_file("suzanne.fbx", True)
print(f"Vertices: {mesh.get_num_vertices()}, Faces: {mesh.get_num_faces()}, Has UVs: {mesh.has_uvs()}")

if not mesh.has_uvs():
    print("Warning: mesh has no UV coordinates. The output will be black.")

builder = CPUBuilder(mesh)
bvh_data = builder.build_bvh(25)
bvh = GPUTraverser(bvh_data)

img_size = 800
n_pixels = img_size * img_size


# ==== Generate rays ==== #

mesh_min, mesh_max = mesh.get_bounds()
max_extent = max(mesh_max - mesh_min)
center = (mesh_max + mesh_min) * 0.5

cam_pos = np.array([
    center[0] + max_extent * 1.0,
    center[1] - max_extent * 1.5,
    center[2] + max_extent * 0.5,
])
cam_poses = np.tile(cam_pos, (n_pixels, 1)).astype(np.float32)
cam_dir = (center - cam_pos) * 0.9

x_dir = np.cross(cam_dir, np.array([0, 0, 1]))
x_dir = x_dir / np.linalg.norm(x_dir) * (max_extent / 2)

y_dir = -np.cross(x_dir, cam_dir)
y_dir = y_dir / np.linalg.norm(y_dir) * (max_extent / 2)

x_coords, y_coords = np.meshgrid(
    np.linspace(-1, 1, img_size),
    np.linspace(-1, 1, img_size),
)

x_coords = x_coords.flatten()
y_coords = y_coords.flatten()

dirs = (cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]).astype(np.float32)

d_cam_poses = torch.from_numpy(cam_poses).cuda()
d_dirs = torch.from_numpy(dirs).cuda()


# ==== Ray trace ==== #

ray_tracer = GPURayTracer(bvh_data)
mask, t, normals, uvs = ray_tracer.trace(d_cam_poses, d_dirs)

print(uvs[uvs.sum(dim=1) != 0])

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
