import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from mesh_utils import Mesh, MeshSamplerMode, GPUMeshSampler
from mesh_utils import GPUTraverser


n_points = 50000

mesh = Mesh.from_file("suzanne.fbx", up_axis="z", forward_axis="y", build_bvh=True, max_leaf_size=25)
bvh_data = mesh.get_bvh()
bvh_data.save_to_obj("bvh.obj", 25)
bvh = GPUTraverser(bvh_data)

sampler = GPUMeshSampler(mesh, MeshSamplerMode.SURFACE_UNIFORM, n_points)

points = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
barycentrics = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
face_idxs = torch.zeros((n_points,), dtype=torch.uint32, device="cuda")

sampler.sample(points, barycentrics, face_idxs, n_points)

print(points)

# create .obj file to visualize the sampled points
with open("sampled_points.obj", "w") as f:
    for p in points.cpu().numpy():
        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")

print(mesh.get_bounds())
mesh.save_to_obj("original_mesh.obj")
mesh.save_preview("original_mesh.png", 800, 800)

t = torch.zeros((n_points,), dtype=torch.float32, device="cuda") + 1e9
closests = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
barycentrics = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
face_idxs2 = torch.zeros((n_points,), dtype=torch.uint32, device="cuda")
bvh.point_query(points, t, closests, barycentrics, face_idxs2)

with open("sampled_points2.obj", "w") as f:
    for p in closests.cpu().numpy():
        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")
