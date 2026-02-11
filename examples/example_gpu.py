import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from mesh_utils import Mesh, GPUTraverser
from mesh_utils import GPURayTracer


# ==== Load mesh with BVH ==== #

mesh = Mesh.from_file("/home/me/Downloads/Untitled.glb", scale=1, build_bvh=True, max_leaf_size=5)
# mesh.split_faces(0.9)

mesh.save_preview("mesh_preview.png", 800, 800)

bvh_data = mesh.get_bvh()
bvh_data.save_to_obj("bvh.obj", 25)
bvh = GPUTraverser(bvh_data)

img_size = 800
n_pixels = img_size * img_size


# ==== Generate rays ==== #

mesh_min, mesh_max = mesh.get_bounds()
max_extent = max(mesh_max - mesh_min)

center = (mesh_max + mesh_min) * 0.5

cam_pos = np.array([
    center[0] + max_extent * 1.0,
    center[1] + max_extent * 0.5,
    center[2] + max_extent * 1.5,
])
cam_poses = np.tile(cam_pos, (n_pixels, 1))
cam_dir = (center - cam_pos) * 0.9

x_dir = np.cross(cam_dir, np.array([0, 1, 0]))
x_dir = x_dir / np.linalg.norm(x_dir) * (max_extent / 2)

y_dir = -np.cross(x_dir, cam_dir)
y_dir = y_dir / np.linalg.norm(y_dir) * (max_extent / 2)

x_coords, y_coords = np.meshgrid(
    np.linspace(-1, 1, img_size),
    np.linspace(-1, 1, img_size),
)

x_coords = x_coords.flatten()
y_coords = y_coords.flatten()

dirs = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]

d_cam_poses = torch.from_numpy(cam_poses).cuda()
d_dirs = torch.from_numpy(dirs).cuda()


# ==== Run BVH ==== #

ray_tracer = GPURayTracer(bvh_data)
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
