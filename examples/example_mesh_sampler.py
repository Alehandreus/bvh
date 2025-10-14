import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from bvh import Mesh, MeshSamplerMode, GPUMeshSampler


n_points = 50000

mesh = Mesh("/home/me/brain/mesh-mapping/monkey_rough_3.fbx")

sampler = GPUMeshSampler(mesh, MeshSamplerMode.SURFACE_UNIFORM, n_points)

points = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")

sampler.sample(points, n_points)

# create .obj file to visualize the sampled points

with open("sampled_points.obj", "w") as f:
    for p in points.cpu().numpy():
        f.write(f"v {p[0]} {p[1]} {p[2]}\n")
