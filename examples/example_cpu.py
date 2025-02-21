import numpy as np
import matplotlib.pyplot as plt

from bvh import BVH


# ==== Load and prepare BVH ==== #

loader = BVH()
loader.load_scene("suzanne.fbx")
loader.build_bvh(25)

img_size = 1000
n_pixels = img_size * img_size


# ==== Generate rays ==== #

mesh_min, mesh_max = loader.mesh_bounds()
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


# ==== Run BVH ==== #

mode = "closest_primitive"

if mode == "closest_primitive":
    mask, t = loader.closest_primitive(cam_poses, dirs)

if mode == "closest_bbox":
    mask, bbox_idxs, t1, t2 = loader.closest_bbox(cam_poses, dirs)
    t = t1

if mode == "random_bbox":
    loader.reset_stack(cam_poses.shape[0])
    alive, mask, bbox_idxs, t1, t2 = loader.another_bbox(cam_poses, dirs)
    t = t1

if mode == "another_bbox":
    mask = np.array([False] * cam_poses.shape[0])
    bbox_idxs = np.zeros((cam_poses.shape[0],), dtype=np.uint32)
    t1 = np.ones((cam_poses.shape[0],)) * 1e9
    t2 = np.ones((cam_poses.shape[0],)) * 1e9

    alive = True
    loader.reset_stack(cam_poses.shape[0])
    while alive:
        alive, cur_mask, cur_bbox_idxs, cur_t1, cur_t2 = loader.another_bbox(cam_poses, dirs)
        mask = mask | cur_mask
        update_mask = cur_mask & (cur_t1 < t1)

        bbox_idxs[update_mask] = cur_bbox_idxs[update_mask]
        t1[update_mask] = cur_t1[update_mask]
        t2[update_mask] = cur_t2[update_mask]

    t = t1
    t[t == 1e9] = 0


# ==== Visualize ==== #

mask_img = mask.reshape(img_size, img_size)

image = np.zeros((img_size, img_size, 3))

img = t.reshape(img_size, img_size)
img[~mask_img] = np.min(img[mask_img])
img = (img - np.min(img)) / (np.max(img) - np.min(img))
img = img * (0.9 - 0.3) + 0.3
img[~mask_img] = 1
img[0, 0] = 0 # I am deeply sorry for this

plt.axis('off')

plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.savefig('output.png')
