import numpy as np
import matplotlib.pyplot as plt
import torch

from bvh import Mesh, CPUBuilder, GPUTraverser
from bvh import TreeType, TraverseMode


# ==== Load and prepare BVH ==== #

mesh = Mesh("suzanne.fbx")
mesh.split_faces(0.9)

builder = CPUBuilder(mesh)
bvh_data = builder.build_bvh(25)
bvh = GPUTraverser(bvh_data)

img_size = 1000
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

depths = torch.zeros((cam_poses.shape[0],), dtype=torch.uint32).cuda()
bbox_idxs = torch.zeros((cam_poses.shape[0], 64), dtype=torch.uint32).cuda()
mask = torch.zeros((cam_poses.shape[0],), dtype=torch.bool).cuda()
t1 = torch.zeros((cam_poses.shape[0],), dtype=torch.float32).cuda() + 1e9
t2 = torch.zeros((cam_poses.shape[0],), dtype=torch.float32).cuda() + 1e9

mode = TraverseMode.CLOSEST_PRIMITIVE

if mode != TraverseMode.ANOTHER_BBOX:
    bvh.traverse(d_cam_poses, d_dirs, mask, t1, t2, depths, bbox_idxs, TreeType.BVH, mode)
else:
    total_mask = torch.zeros((cam_poses.shape[0],), dtype=torch.bool).cuda()
    total_t = torch.zeros((cam_poses.shape[0],), dtype=torch.float32).cuda() + 1e9

    alive = True
    bvh.reset_stack(cam_poses.shape[0])
    while alive:
        alive = bvh.traverse(d_cam_poses, d_dirs, depths, bbox_idxs, bbox_idxs, mask, t1, t2, False)

        total_mask = total_mask | mask
        total_t[mask & (t1 < total_t)] = t1[mask & (t1 < total_t)]

    mask = total_mask
    t1 = total_t
    t1[t1 == 1e9] = 0


# ==== Visualize ==== #

mask_img = mask.reshape(img_size, img_size)
mask_img = mask_img.cpu().numpy()
t = t1.cpu().numpy()

image = np.zeros((img_size, img_size, 3))

img = t.reshape(img_size, img_size)
img[~mask_img] = np.min(img[mask_img])
img = (img - np.min(img)) / (np.max(img) - np.min(img))
img = 1 - img
img[~mask_img] = 0

plt.axis('off')
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.savefig('output.png', bbox_inches='tight', pad_inches=0)
