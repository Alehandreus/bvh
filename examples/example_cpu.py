import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from mesh_utils import Mesh, CPUBuilder, CPUTraverser


# ==== Load and prepare BVH ==== #

mesh = Mesh.from_file("suzanne.fbx")
# mesh.split_faces(0.9)

builder = CPUBuilder(mesh)
bvh_data = builder.build_bvh(25)
bvh_data.save_to_obj("bvh.obj", 25)
bvh = CPUTraverser(bvh_data)

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
cam_poses = np.tile(cam_pos, (n_pixels, 1))
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

dirs = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]


# ==== Run BVH ==== #

depths = np.zeros((cam_poses.shape[0],), dtype=np.uint32)
prim_idxs = np.zeros((cam_poses.shape[0],), dtype=np.uint32)
mask = np.zeros((cam_poses.shape[0],), dtype=np.bool_)
t = np.zeros((cam_poses.shape[0],), dtype=np.float32) + 1e9
normals = np.zeros((cam_poses.shape[0], 3), dtype=np.float32)
uvs = np.zeros((cam_poses.shape[0], 2), dtype=np.float32)

bvh.ray_query(cam_poses, dirs, mask, t, prim_idxs, normals, uvs)

# ==== Visualize ==== #

mask_img = mask.reshape(img_size, img_size)

light_dir = np.array([1, -1, 1])
light_dir = light_dir / np.linalg.norm(light_dir)

normals[np.isnan(normals)] = 0
colors = np.dot(normals, light_dir) * 0.5 + 0.5

img = colors.reshape(img_size, img_size)
img[~mask_img] = 0

image = Image.fromarray((img * 255).astype(np.uint8))
image.save('output.png')
