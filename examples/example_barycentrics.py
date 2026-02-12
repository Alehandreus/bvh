import numpy as np
import torch
from PIL import Image

from mesh_utils import Mesh, GPURayTracer, generate_camera_rays


# ==== Build mesh + ray tracer ==== #

mesh = Mesh.from_file("/home/me/Downloads/shrek.glb", build_bvh=True, max_leaf_size=25)
print(f"Vertices: {mesh.get_num_vertices()}, Faces: {mesh.get_num_faces()}")

img_size = 800

# Each vertex gets a random color, consistent across the mesh
vertex_colors = torch.rand((mesh.get_num_vertices(), 3), dtype=torch.float32)
faces = mesh.get_faces()
print(faces.shape)


# ==== Generate rays and trace ==== #

d_cam_poses, d_dirs = generate_camera_rays(mesh, img_size)

ray_tracer = GPURayTracer(mesh)
result = ray_tracer.trace(d_cam_poses, d_dirs)
t = result.distance.cpu().numpy()
mask = result.mask.cpu().numpy()
normals = result.normals.cpu().numpy()
barycentrics = result.barycentrics.cpu().numpy()
face_idx = result.face_idx.cpu().numpy()


# ==== Interpolate vertex colors per hit using barycentrics ==== #

vertex_colors = vertex_colors.cpu().numpy()
colors = np.zeros((img_size * img_size, 3), dtype=np.float32)
hit_faces = faces[face_idx[mask]]
hit_bary = barycentrics[mask]

colors[mask] = (
    hit_bary[:, 0:1] * vertex_colors[hit_faces[:, 0]]
    + hit_bary[:, 1:2] * vertex_colors[hit_faces[:, 1]]
    + hit_bary[:, 2:3] * vertex_colors[hit_faces[:, 2]]
)

# Simple lighting to give a bit of shading
light_dir = np.array([1.0, -1.0, 1.0], dtype=np.float32)
light_dir /= np.linalg.norm(light_dir)
lighting = np.clip(np.einsum("ij,j->i", normals, light_dir) * 0.5 + 0.5, 0.0, 1.0)
colors *= lighting[:, None]
colors[~mask] = 0


# ==== Save result ==== #

img = colors.reshape(img_size, img_size, 3)
img = np.clip(img * 255, 0, 255).astype(np.uint8)
Image.fromarray(img).save("output.png")
print("Saved output_vertex_colors.png")
