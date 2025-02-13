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
loader.load_scene("suzanne2.fbx")
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

start = origins + directions * 0.3
end = origins + directions * 0.7

n_segments = 64
segments = loader.intersect_segments(start, end, n_segments)

segments_argmax = np.argmax(segments, axis=1)

segments_argmax = segments_argmax.reshape(resolution, resolution)

img = color_pool[segments_argmax % len(color_pool)]
img[segments_argmax == 0] = 255

plt.axis('off')
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.savefig('suzanne.png')
plt.show()
