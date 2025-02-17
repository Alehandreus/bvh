import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from bvh import BVH


def cut_edges(img):    
    edge = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    edge_img = np.maximum(convolve2d(img, edge, mode='same'), 0) > 0

    mixed = (1 - edge_img) * img
    return mixed


# blue, yellow, pink, whatever man, just keep bringing me that
color_pool = np.array([
    [  0, 128, 128],  # Teal Blue
    [255,  94,  77],  # Sunset Orange
    [120,  81, 169],  # Royal Purple
    [218, 165,  32],  # Goldenrod
    [152, 255, 152],  # Mint Green
    [255, 127, 139],  # Coral Pink
    [  0, 191, 255],  # Deep Sky Blue
    [220,  20,  60],  # Crimson Red
    [204, 255,   0],  # Electric Lime
    [244, 196,  48]   # Saffron Yellow
])


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

stack_size = np.ones((origins.shape[0],), dtype=np.int32)
stack = np.zeros((origins.shape[0], 20), dtype=np.uint32)
mask, leaf_indices, t1, t2 = loader.intersect_leaves(origins, directions, stack_size, stack)
mask_img = mask.reshape(resolution, resolution)

image = np.zeros((resolution, resolution, 3))

print("Number of nodes: ", loader.n_nodes)
print("Number of leaves: ", loader.n_leaves)

# ==== outline ====
outline = leaf_indices.reshape(resolution, resolution)
outline = cut_edges(outline)

# ==== color ====
img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
img[mask_img] = color_pool[leaf_indices[mask] % len(color_pool)]
img = img * (outline[:, :, None] > 0)
img[~mask_img] = 255

# ==== grayscale ====
# img = t1.reshape(resolution, resolution)
# img = (img - np.min(img)) / (np.max(img) - np.min(img))
# img = img * 0.8
# img = cut_edges(img)
# img[~mask_img] = 1

plt.axis('off')

plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.savefig('suzanne2.png')
# plt.show()
