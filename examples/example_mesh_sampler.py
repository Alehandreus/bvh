import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from mesh_utils import Mesh, MeshSampler, GPUTraverser


n_points = 50000

mesh = Mesh.from_file("suzanne.fbx", up_axis="z", forward_axis="y", build_bvh=True, max_leaf_size=25)
bvh = GPUTraverser(mesh)

sampler = MeshSampler(mesh, max_points=n_points)
result = sampler.sample(n_points)

print(f"Sampled {n_points} points")
print(f"Points shape: {result.points.shape}")

# create .obj file to visualize the sampled points
with open("sampled_points.obj", "w") as f:
    for p in result.points.cpu().numpy():
        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")

mesh.save_to_obj("original_mesh.obj")
mesh.save_preview("original_mesh.png")

t = torch.zeros((n_points,), dtype=torch.float32, device="cuda") + 1e9
closests = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
barycentrics = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
face_idxs2 = torch.zeros((n_points,), dtype=torch.uint32, device="cuda")
bvh.point_query(result.points, t, closests, barycentrics, face_idxs2)

with open("sampled_points2.obj", "w") as f:
    for p in closests.cpu().numpy():
        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")
