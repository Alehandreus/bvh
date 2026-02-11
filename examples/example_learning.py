import torch
from torch import nn
from torch.nn import functional as F
import tqdm
import numpy as np
from PIL import Image

from mesh_utils import Mesh, GPURayTracer, generate_camera_rays

"""

Neural Network training example for predicting vertex colors
by encoding points using learnable vertex embeddings and a small MLP.

"""


class Model(nn.Module):
    def __init__(self, mesh, emb_size=16):
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = nn.Embedding(mesh.get_num_vertices(), emb_size)
        self.faces = nn.Parameter(torch.from_numpy(mesh.get_faces()).long(), requires_grad=False)

        self.net = nn.Sequential(
            nn.Linear(emb_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

    def forward(self, barycentrics, face_idx):
        v0_idx = self.faces[face_idx, 0]
        v1_idx = self.faces[face_idx, 1]
        v2_idx = self.faces[face_idx, 2]

        emb_v0 = self.embeddings(v0_idx)
        emb_v1 = self.embeddings(v1_idx)
        emb_v2 = self.embeddings(v2_idx)

        w0 = barycentrics[:, 0]
        w1 = barycentrics[:, 1]
        w2 = barycentrics[:, 2]

        point_emb = w0[:, None] * emb_v0 + w1[:, None] * emb_v1 + w2[:, None] * emb_v2

        output = self.net(point_emb)

        return output
    

def generate_data(mesh, raytracer, n_rays=1000):
    # generate randomly origin inside bounding box and random direction
    bbox_min, bbox_max = mesh.get_bounds()
    bbox_min = torch.from_numpy(bbox_min).float().cuda()
    bbox_max = torch.from_numpy(bbox_max).float().cuda()

    origins = torch.rand((n_rays, 3)).float().cuda() * (bbox_max - bbox_min) + bbox_min
    directions = torch.randn((n_rays, 3)).float().cuda()
    directions = F.normalize(directions, dim=1)

    # remove rays that don't hit the mesh
    result = raytracer.trace(origins, directions)
    mask = result.mask

    return result.barycentrics[mask], result.face_idx[mask], result.color[mask]

    

if __name__ == "__main__":
    num_epochs = 100000
    batch_size = 100000
    val_every = 1000
    img_size = 1024

    mesh = Mesh.from_file("/home/me/Downloads/shrek.glb", build_bvh=True)
    ray_tracer = GPURayTracer(mesh)

    model = Model(mesh).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in (bar := tqdm.tqdm(range(num_epochs))):
        model.train()
        barycentrics, face_idx, gt_color = generate_data(mesh, ray_tracer, n_rays=10000)
        pred_color = model(barycentrics, face_idx)
        loss = F.mse_loss(pred_color, gt_color)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.set_description(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        if epoch % val_every == 0:
            with torch.no_grad():
                model.eval()
                origs, directions = generate_camera_rays(mesh, img_size=img_size, device="cuda")
                result = ray_tracer.trace(origs, directions)
                mask = result.mask
                
                pred_colors = model(result.barycentrics[mask], result.face_idx[mask])

                img = torch.zeros((img_size * img_size, 3), device="cuda")
                img[mask] = pred_colors
                img = img.cpu().numpy().reshape((img_size, img_size, 3))
                img = (img * 255).astype(np.uint8)
                Image.fromarray(img).save(f"output.png")

