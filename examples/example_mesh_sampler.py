import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from bvh import Mesh, MeshSamplerMode, GPUMeshSampler


n_points = 50000

mesh = Mesh.from_file("/home/me/brain/mesh-mapping/models/queen_fine.fbx")

sampler = GPUMeshSampler(mesh, MeshSamplerMode.SURFACE_UNIFORM, n_points)

points = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")

sampler.sample(points, n_points)

# create .obj file to visualize the sampled points

with open("sampled_points.obj", "w") as f:
    for p in points.cpu().numpy():
        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")

v = mesh.get_vertices()
f = mesh.get_faces()
print(mesh.get_bounds())
mesh = Mesh.from_data(v, f)
mesh.save_to_obj("original_mesh.obj")
mesh.save_preview("original_mesh.png", 800, 800)