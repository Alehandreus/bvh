import torch
import numpy as np
from collections import namedtuple
from .mesh_utils_impl import Mesh, GPUTraverser, GPUMeshSampler

RayQueryResult = namedtuple('RayQueryResult', ['mask', 'distance', 'normals', 'uv', 'color', 'face_idx'])
SampleResult = namedtuple('SampleResult', ['points', 'barycentrics', 'face_indices'])


def generate_camera_rays(mesh, img_size=512, device=None):
    """Generate camera rays for rendering a mesh.

    Automatically positions the camera to view the entire mesh from
    an upper-right viewpoint using PyTorch so the outputs can be pushed
    straight to a GPU.

    Args:
        mesh: Mesh object to generate rays for
        img_size: Image resolution (default 512x512)
        device: Torch device (string or torch.device). Defaults to CUDA if
            available, otherwise CPU.

    Returns:
        cam_poses: (img_size*img_size, 3) tensor of ray origins
        dirs: (img_size*img_size, 3) tensor of ray directions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n_pixels = img_size * img_size

    mesh_min, mesh_max = mesh.get_bounds()
    mesh_min = torch.as_tensor(mesh_min, dtype=torch.float32, device=device)
    mesh_max = torch.as_tensor(mesh_max, dtype=torch.float32, device=device)

    max_extent = torch.max(mesh_max - mesh_min)
    center = (mesh_max + mesh_min) * 0.5

    cam_pos = torch.stack([
        center[0] + max_extent * 1.0,
        center[1] + max_extent * 0.5,
        center[2] + max_extent * 1.5,
    ])
    cam_dir = (center - cam_pos) * 0.9

    camera_up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    x_dir = torch.linalg.cross(cam_dir, camera_up)
    x_dir = x_dir / torch.linalg.norm(x_dir) * (max_extent / 2)

    y_dir = -torch.linalg.cross(x_dir, cam_dir)
    y_dir = y_dir / torch.linalg.norm(y_dir) * (max_extent / 2)

    x_coords = torch.linspace(-1, 1, img_size, device=device)
    y_coords = torch.linspace(-1, 1, img_size, device=device)
    x_flat = x_coords.repeat(img_size)
    y_flat = y_coords.repeat_interleave(img_size)

    cam_poses = cam_pos.unsqueeze(0).repeat(n_pixels, 1)
    dirs = cam_dir.unsqueeze(0) + x_dir.unsqueeze(0) * x_flat.unsqueeze(1) + y_dir.unsqueeze(0) * y_flat.unsqueeze(1)

    return cam_poses, dirs


class GPURayTracer:
    def __init__(self, mesh):
        """Initialize GPU ray tracer.

        Args:
            mesh: Mesh object to trace rays against (must have BVH built)
        """
        self.bvh_traverser = GPUTraverser(mesh)

    def trace(self, cam_poses, dirs, allow_negative=True, allow_backward=True, allow_forward=True):
        """Trace rays through the mesh.

        Args:
            cam_poses: (n_rays, 3) tensor of ray origins
            dirs: (n_rays, 3) tensor of ray directions
            allow_negative: Allow negative ray distances
            allow_backward: Allow hits behind ray origin
            allow_forward: Allow hits in front of ray origin

        Returns:
            RayQueryResult with:
                mask: (n_rays,) bool tensor of hits
                distance: (n_rays,) float tensor of hit distances
                normals: (n_rays, 3) float tensor of surface normals
                uv: (n_rays, 2) float tensor of UV coordinates
                color: (n_rays, 3) float tensor of surface colors
                face_idx: (n_rays,) uint32 tensor of the hit face index (undefined for misses)
        """
        n_rays = cam_poses.shape[0]

        face_idxs = torch.zeros((n_rays,),    dtype=torch.uint32,  device="cuda")
        mask      = torch.zeros((n_rays,),    dtype=torch.bool,    device="cuda")
        t         = torch.zeros((n_rays,),    dtype=torch.float32, device="cuda") + 1e9
        normals   = torch.zeros((n_rays, 3),  dtype=torch.float32, device="cuda")
        uvs       = torch.zeros((n_rays, 2),  dtype=torch.float32, device="cuda")
        colors    = torch.zeros((n_rays, 3),  dtype=torch.float32, device="cuda")

        self.bvh_traverser.ray_query(
            cam_poses,
            dirs,
            mask,
            t,
            face_idxs,
            normals,
            uvs,
            colors,
            allow_negative,
            allow_backward,
            allow_forward,
        )

        return RayQueryResult(mask=mask, distance=t, normals=normals, uv=uvs, color=colors, face_idx=face_idxs.long())


class MeshSampler:
    def __init__(self, mesh, max_points=100000):
        """Initialize mesh sampler.

        Args:
            mesh: Mesh object to sample from
            max_points: Maximum number of points that can be sampled
        """
        self.sampler = GPUMeshSampler(mesh, max_points)
        self.max_points = max_points

    def sample(self, n_points):
        """Sample points on the mesh surface.

        Args:
            n_points: Number of points to sample

        Returns:
            SampleResult with:
                points: (n_points, 3) tensor of sampled positions
                barycentrics: (n_points, 3) tensor of barycentric coordinates
                face_indices: (n_points,) tensor of face indices
        """
        if n_points > self.max_points:
            raise ValueError(f"n_points ({n_points}) exceeds max_points ({self.max_points})")

        # Allocate output tensors
        points = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
        barycentrics = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
        face_indices = torch.zeros((n_points,), dtype=torch.uint32, device="cuda")

        # Sample
        self.sampler.sample(points, barycentrics, face_indices, n_points)

        return SampleResult(points=points, barycentrics=barycentrics, face_indices=face_indices)
