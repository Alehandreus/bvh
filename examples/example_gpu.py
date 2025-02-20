import numpy as np
import matplotlib.pyplot as plt
import torch

from bvh import BVH

loader = BVH()
loader.load_scene("suzanne.fbx")
loader.build_bvh(15)
loader.cudify()

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

origins = torch.tensor(origins, dtype=torch.float32, device='cuda')
directions = torch.tensor(directions, dtype=torch.float32, device='cuda')

mode = "closest_primitive"

if mode == "closest_primitive":
    mask, t = loader.closest_primitive_cuda(origins, directions)

mask_img = mask.reshape(resolution, resolution)
mask_img = mask_img.cpu().numpy()
t = t.cpu().numpy()

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
