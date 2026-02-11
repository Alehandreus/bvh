import torch
import numpy as np
from collections import namedtuple
from .mesh_utils_impl import Mesh, CPUTraverser, GPUTraverser
from .mesh_utils_impl import GPUMeshSampler, MeshSamplerMode

RayQueryResult = namedtuple('RayQueryResult', ['mask', 'distance', 'normals', 'uv', 'color'])
SampleResult = namedtuple('SampleResult', ['points', 'barycentrics', 'face_indices'])


def generate_camera_rays(mesh, img_size=512):
    """Generate camera rays for rendering a mesh.

    Automatically positions the camera to view the entire mesh from
    an upper-right viewpoint.

    Args:
        mesh: Mesh object to generate rays for
        img_size: Image resolution (default 800x800)

    Returns:
        cam_poses: (img_size*img_size, 3) array of ray origins
        dirs: (img_size*img_size, 3) array of ray directions
    """
    n_pixels = img_size * img_size

    # Get mesh bounds
    mesh_min, mesh_max = mesh.get_bounds()
    max_extent = max(mesh_max - mesh_min)
    center = (mesh_max + mesh_min) * 0.5

    # Position camera above and to the right
    cam_pos = np.array([
        center[0] + max_extent * 1.0,
        center[1] + max_extent * 0.5,
        center[2] + max_extent * 1.5,
    ])
    cam_poses = np.tile(cam_pos, (n_pixels, 1))
    cam_dir = (center - cam_pos) * 0.9

    # Calculate image plane basis vectors (Y-up coordinate system)
    x_dir = np.cross(cam_dir, np.array([0, 1, 0]))
    x_dir = x_dir / np.linalg.norm(x_dir) * (max_extent / 2)

    y_dir = -np.cross(x_dir, cam_dir)
    y_dir = y_dir / np.linalg.norm(y_dir) * (max_extent / 2)

    # Generate pixel coordinates
    x_coords, y_coords = np.meshgrid(
        np.linspace(-1, 1, img_size),
        np.linspace(-1, 1, img_size),
    )
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    # Generate ray directions
    dirs = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]

    return cam_poses.astype(np.float32), dirs.astype(np.float32)


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
        """
        n_rays = cam_poses.shape[0]

        prim_idxs = torch.zeros((n_rays,),    dtype=torch.uint32,  device="cuda")
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
            prim_idxs,
            normals,
            uvs,
            colors,
            allow_negative,
            allow_backward,
            allow_forward,
        )

        return RayQueryResult(mask=mask, distance=t, normals=normals, uv=uvs, color=colors)

class MeshSampler:
    def __init__(self, mesh, mode=None, max_points=100000):
        """Initialize mesh sampler.

        Args:
            mesh: Mesh object to sample from
            mode: Sampling mode (default: SURFACE_UNIFORM)
            max_points: Maximum number of points that can be sampled
        """
        if mode is None:
            mode = MeshSamplerMode.SURFACE_UNIFORM
        self.sampler = GPUMeshSampler(mesh, mode, max_points)
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
