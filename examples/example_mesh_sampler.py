import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from mesh_utils import Mesh, MeshSamplerMode, GPUMeshSampler
from mesh_utils import GPUTraverser, CPUBuilder


n_points = 50000

mesh = Mesh.from_file("suzanne.fbx")

builder = CPUBuilder(mesh)
bvh_data = builder.build_bvh(25)
bvh_data.save_to_obj("bvh.obj", 25)
bvh = GPUTraverser(bvh_data)

mesh = Mesh.from_data(bvh_data.get_vertices(), bvh_data.get_faces())

sampler = GPUMeshSampler(mesh, MeshSamplerMode.SURFACE_UNIFORM, n_points)

points = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
barycentrics = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
face_idxs = torch.zeros((n_points,), dtype=torch.uint32, device="cuda")

sampler.sample(points, barycentrics, face_idxs, n_points)

# create .obj file to visualize the sampled points
with open("sampled_points.obj", "w") as f:
    for p in points.cpu().numpy():
        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")

v = mesh.get_vertices()
f = mesh.get_faces()
print(mesh.get_bounds())
mesh = Mesh.from_data(v, f)
mesh.save_to_obj("original_mesh.obj")
mesh.save_preview("original_mesh.png", 800, 800, mesh.get_c(), mesh.get_R())

t = torch.zeros((n_points,), dtype=torch.float32, device="cuda") + 1e9
closests = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
barycentrics = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
face_idxs2 = torch.zeros((n_points,), dtype=torch.uint32, device="cuda")
bvh.point_query(points, t, closests, barycentrics, face_idxs2)

with open("sampled_points2.obj", "w") as f:
    for p in closests.cpu().numpy():
        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")
