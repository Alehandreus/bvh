import torch

from mesh_utils import Mesh, MeshSampler


n_points = 50000

mesh = Mesh.from_file("suzanne.fbx", up_axis="y", forward_axis="-z", build_bvh=True)

sampler = MeshSampler(mesh, max_points=n_points)
result = sampler.sample(n_points)

print(f"Sampled {n_points} points")
print(f"Points shape: {result.points.shape}")

# create .obj file to visualize the sampled points
with open("sampled_points.obj", "w") as f:
    for p in result.points.cpu().numpy():
        f.write(f"v {p[0]} {p[1]} {p[2]}\n")

mesh.save_to_obj("original_mesh.obj")
mesh.save_preview("output.png")
