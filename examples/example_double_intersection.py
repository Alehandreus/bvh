import numpy as np
import matplotlib.pyplot as plt
import torch

from mesh_utils import Mesh, GPURayTracer, generate_camera_rays

mesh = Mesh.from_file("/home/me/brain/mesh-mapping/models/dragon_outer_2000.fbx", up_axis="z", forward_axis="y", build_bvh=True, max_leaf_size=5)
tracer = GPURayTracer(mesh)

img_size = 800

cam_poses, dirs = generate_camera_rays(mesh, img_size)
dirs_normalized = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

d_cam_poses = torch.from_numpy(cam_poses).float().cuda()
d_dirs = torch.from_numpy(dirs).float().cuda()
d_dirs_normalized = torch.from_numpy(dirs_normalized).float().cuda()

# First intersection
mask_1, t_1, normals_1, uvs_1 = tracer.trace(d_cam_poses, d_dirs, allow_negative=False)

# Second intersection - move epsilon past first hit and trace again
epsilon = 1e-3
first_hit_points = d_cam_poses + d_dirs * t_1[:, None]
new_ray_origins = first_hit_points + d_dirs * epsilon

mask_2, t_2, normals_2, uvs_2 = tracer.trace(new_ray_origins, d_dirs, allow_negative=False)

# Visualize
mask_1_img = mask_1.reshape(img_size, img_size).cpu().numpy()
mask_2_img = mask_2.reshape(img_size, img_size).cpu().numpy()
normals_1_np = normals_1.cpu().numpy()
normals_2_np = normals_2.cpu().numpy()
dirs_np = d_dirs_normalized.cpu().numpy()

light_dir = np.array([1, -1, 1])
light_dir = light_dir / np.linalg.norm(light_dir)

normals_1_np[np.isnan(normals_1_np)] = 0
normals_2_np[np.isnan(normals_2_np)] = 0

# Flip back-facing normals for proper shading
dot_1_np = np.sum(dirs_np * normals_1_np, axis=1)
dot_2_np = np.sum(dirs_np * normals_2_np, axis=1)

normals_1_shading = normals_1_np.copy()
normals_1_shading[dot_1_np > 0] *= -1

normals_2_shading = normals_2_np.copy()
normals_2_shading[dot_2_np > 0] *= -1

colors_1 = np.abs(np.dot(normals_1_shading, light_dir))
colors_2 = np.abs(np.dot(normals_2_shading, light_dir))

img_1 = colors_1.reshape(img_size, img_size)
img_1[~mask_1_img] = 0

img_2 = colors_2.reshape(img_size, img_size)
img_2[~mask_2_img] = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(img_1, cmap='gray')
ax1.set_title('First Intersection (Entry)')
ax1.axis('off')

ax2.imshow(img_2, cmap='gray')
ax2.set_title('Second Intersection (Exit)')
ax2.axis('off')

plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
print(f"Rendered {mask_1.sum().item()} entry points, {mask_2.sum().item()} exit points")
