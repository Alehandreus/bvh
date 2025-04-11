import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from bvh import Mesh, CPUBuilder, GPUTraverser
from bvh import TreeType, TraverseMode


# ==== Load and prepare BVH ==== #

# mesh = Mesh("suzanne.fbx")
mesh = Mesh("models/lego.fbx")
# mesh.split_faces(0.9)

builder = CPUBuilder(mesh)
bvh_data = builder.build_bvh(5)
bvh_data.save_as_obj("bvh.obj", 25)
bvh = GPUTraverser(bvh_data)

img_size = 4096
n_pixels = img_size * img_size


# ==== Generate rays ==== #

mesh_min, mesh_max = mesh.bounds()
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

d_cam_poses = torch.from_numpy(cam_poses).cuda()
d_dirs = torch.from_numpy(dirs).cuda()


# ==== Run BVH ==== #

depths = torch.zeros((cam_poses.shape[0],), dtype=torch.int).cuda()
bbox_idxs = torch.zeros((cam_poses.shape[0]), dtype=torch.uint32).cuda()
history = torch.zeros((cam_poses.shape[0], 64), dtype=torch.uint32).cuda()
mask = torch.zeros((cam_poses.shape[0],), dtype=torch.bool).cuda()
t1 = torch.zeros((cam_poses.shape[0],), dtype=torch.float32).cuda() + 1e9
t2 = torch.zeros((cam_poses.shape[0],), dtype=torch.float32).cuda() + 1e9
normals = torch.zeros((cam_poses.shape[0], 3), dtype=torch.float32).cuda()

mode = TraverseMode.CLOSEST_PRIMITIVE

if mode != TraverseMode.ANOTHER_BBOX:
    bvh.traverse(d_cam_poses, d_dirs, mask, t1, t2, bbox_idxs, normals, TreeType.BVH, mode)
else:
    total_mask = torch.zeros((cam_poses.shape[0],), dtype=torch.bool).cuda()
    total_t = torch.zeros((cam_poses.shape[0],), dtype=torch.float32).cuda() + 1e9

    alive = True
    bvh.reset_stack(cam_poses.shape[0])
    while alive:
        alive = bvh.traverse(d_cam_poses, d_dirs, mask, t1, t2, bbox_idxs, normals, False)

        total_mask = total_mask | mask
        total_t[mask & (t1 < total_t)] = t1[mask & (t1 < total_t)]

    mask = total_mask
    t1 = total_t
    t1[t1 == 1e9] = 0


# ==== Visualize ==== #

mask_img = mask.reshape(img_size, img_size)
mask_img = mask_img.cpu().numpy()
normals = normals.cpu().numpy()
t1 = t1.cpu().numpy()

if mode != TraverseMode.CLOSEST_PRIMITIVE:
    img = t1.reshape(img_size, img_size)
    img[~mask_img] = np.min(img[mask_img])
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = 1 - img
    img[~mask_img] = 0

if mode == TraverseMode.CLOSEST_PRIMITIVE:
    light_dir = np.array([1, -1, 1])
    light_dir = light_dir / np.linalg.norm(light_dir)

    normals[np.isnan(normals)] = 0
    colors = np.dot(normals, light_dir) * 0.5 + 0.5
    
    img = colors.reshape(img_size, img_size)
    img[~mask_img] = 0

image = Image.fromarray((img * 255).astype(np.uint8))
image.save('output.png')
