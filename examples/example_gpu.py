import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from mesh_utils import Mesh, GPURayTracer, generate_camera_rays


# ==== Load mesh and buld BVH ==== #

mesh = Mesh.from_file("/home/me/Downloads/shrek.glb", build_bvh=True)
print(f"Mesh loaded: {mesh.get_num_vertices()} vertices, {mesh.get_num_faces()} faces")

# quick rasterization to visualize the mesh
mesh.save_preview("mesh_preview.png")

# ==== Raytrace ==== #

img_size = 800
d_cam_poses, d_dirs = generate_camera_rays(mesh, img_size, distance=1.5, device="cuda")
ray_tracer = GPURayTracer(mesh)
result = ray_tracer.trace(d_cam_poses, d_dirs)
mask = result.mask
distance = result.distance
normals = result.normals
colors = result.color
y = d_cam_poses + d_dirs * distance[:, None]


# ==== Shade and Save ==== #

mask_img = mask.reshape(img_size, img_size)
mask_img = mask_img.cpu().numpy()
normals = normals.cpu().numpy()
distance = distance.cpu().numpy()

light_dir = np.array([1, -1, 1])
light_dir = light_dir / np.linalg.norm(light_dir)

normals[np.isnan(normals)] = 0
colors = colors.cpu().numpy() * (np.dot(normals, light_dir)[:, None] * 0.5 + 0.5)

img = colors.reshape(img_size, img_size, 3)
img[~mask_img] = 0

image = Image.fromarray((img * 255).astype(np.uint8))
image.save('output.png')
