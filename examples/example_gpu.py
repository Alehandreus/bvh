import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from mesh_utils import Mesh, GPURayTracer, generate_camera_rays


# ==== Load mesh with BVH ==== #

mesh = Mesh.from_file("/home/me/Downloads/Untitled.glb", scale=1, build_bvh=True, max_leaf_size=5)

print(f"Mesh loaded: {mesh.get_num_vertices()} vertices, {mesh.get_num_faces()} faces")

mesh.save_preview("mesh_preview.png")

img_size = 800

# ==== Generate rays ==== #

cam_poses, dirs = generate_camera_rays(mesh, img_size)

d_cam_poses = torch.from_numpy(cam_poses).cuda()
d_dirs = torch.from_numpy(dirs).cuda()


# ==== Run BVH ==== #

ray_tracer = GPURayTracer(mesh)
mask, t1, normals, uvs, colors = ray_tracer.trace(d_cam_poses, d_dirs, allow_backward=False, allow_forward=True)
t1[t1 == 1e9] = 0


y = d_cam_poses + d_dirs * t1[:, None]

# ==== Visualize ==== #

mask_img = mask.reshape(img_size, img_size)
mask_img = mask_img.cpu().numpy()
normals = normals.cpu().numpy()
t1 = t1.cpu().numpy()

light_dir = np.array([1, -1, 1])
light_dir = light_dir / np.linalg.norm(light_dir)

normals[np.isnan(normals)] = 0
# colors = colors.cpu().numpy() * np.maximum(np.dot(normals, light_dir)[:, None], 0)
colors = colors.cpu().numpy() * 0 + np.dot(normals, light_dir)[:, None] * 0.5 + 0.5

img = colors.reshape(img_size, img_size, 3)
img[~mask_img] = 0

image = Image.fromarray((img * 255).astype(np.uint8))
image.save('output.png')
