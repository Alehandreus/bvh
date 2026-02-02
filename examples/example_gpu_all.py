import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from mesh_utils import Mesh, CPUBuilder, GPUTraverser
from mesh_utils import GPURayTracer, GPURayTracerAll


# ==== Load and prepare BVH ==== #

fine_mesh = Mesh.from_file("/home/me/brain/mesh-mapping/models/monkey_fine.fbx")
fine_builder = CPUBuilder(fine_mesh)
fine_bvh_data = fine_builder.build_bvh(5)
fine_bvh_data.save_to_obj("bvh.obj", 25)
fine_bvh = GPUTraverser(fine_bvh_data)

rough_mesh = Mesh.from_file("/home/me/brain/mesh-mapping/models/monkey_rough.fbx")
rough_builder = CPUBuilder(rough_mesh)
rough_bvh_data = rough_builder.build_bvh(5)
rough_bvh_data.save_to_obj("bvh_rough.obj", 25)
rough_bvh = GPUTraverser(rough_bvh_data)

img_size = 800
n_pixels = img_size * img_size


# ==== Generate rays ==== #

mesh_min, mesh_max = fine_mesh.get_bounds()
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

fine_ray_tracer = GPURayTracer(fine_bvh_data)
fine_mask, fine_t1, fine_normals, fine_uvs = fine_ray_tracer.trace(d_cam_poses, d_dirs)

rough_ray_tracer_all = GPURayTracerAll(rough_bvh_data)
max_hits_per_ray = 10

mask, t1, prim_idxs, n_hits = rough_ray_tracer_all.trace_all(d_cam_poses, d_dirs, max_hits_per_ray)

t1_expanded = t1[:, :max_hits_per_ray]
fine_t1_expanded = fine_t1[:, None]
diff = torch.abs(t1_expanded - fine_t1_expanded)
min_indices = torch.argmin(diff, dim=1)

mask = mask[:, 0]
min_indices[mask] = min_indices[mask] + 1
min_indices[~mask] = 0


# ==== Visualize ==== #

img = min_indices.reshape(img_size, img_size) / n_hits.max()
img = img.cpu().numpy()
image = Image.fromarray((img * 255).astype(np.uint8))
image.save('output.png')
