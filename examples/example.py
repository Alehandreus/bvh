import numpy as np
import matplotlib.pyplot as plt

from bvh import BVH

loader = BVH()
loader.load_scene("suzanne.fbx")
loader.build_bvh(15)

resolution = 1000

origin = np.array([-1, -5, 0])
pixels = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(1, -1, resolution))
pixels = np.array(pixels).reshape(2, -1).T * 1.5
pixels = np.hstack((
    pixels[:, 0:1],
    np.zeros((pixels.shape[0], 1)),
    pixels[:, 1:2],
))
origins = np.tile(origin, (pixels.shape[0], 1))
directions = pixels - origins

mode = "closest_primitive"

if mode == "closest_primitive":
    mask, t = loader.closest_primitive(origins, directions)
    mask_img = mask.reshape(resolution, resolution)

if mode == "closest_bbox":
    mask, bbox_idxs, t1, t2 = loader.closest_bbox(origins, directions)
    mask_img = mask.reshape(resolution, resolution)
    t = t1

if mode == "random_bbox":
    loader.reset_stack(origins.shape[0])
    alive, mask, bbox_idxs, t1, t2 = loader.another_bbox(origins, directions)
    mask_img = mask.reshape(resolution, resolution)
    t = t1

if mode == "another_bbox":
    mask = np.array([False] * origins.shape[0])
    bbox_idxs = np.zeros((origins.shape[0],), dtype=np.uint32)
    t1 = np.ones((origins.shape[0],)) * 1e9
    t2 = np.ones((origins.shape[0],)) * 1e9

    alive = True
    loader.reset_stack(origins.shape[0])
    while alive:
        alive, cur_mask, cur_bbox_idxs, cur_t1, cur_t2 = loader.another_bbox(origins, directions)
        mask = mask | cur_mask
        update_mask = cur_mask & (cur_t1 < t1)

        bbox_idxs[update_mask] = cur_bbox_idxs[update_mask]
        t1[update_mask] = cur_t1[update_mask]
        t2[update_mask] = cur_t2[update_mask]

    mask_img = mask.reshape(resolution, resolution)
    t = t1
    t[t == 1e9] = 0


image = np.zeros((resolution, resolution, 3))

img = t.reshape(resolution, resolution)
img[~mask_img] = np.min(img[mask_img])
img = (img - np.min(img)) / (np.max(img) - np.min(img))
img = img * (0.9 - 0.3) + 0.3
img[~mask_img] = 1
img[0, 0] = 0 # I am deeply sorry for this

plt.axis('off')

plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.savefig('output.png')
