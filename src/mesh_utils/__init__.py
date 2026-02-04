import torch
from .mesh_utils_impl import Mesh, BVHData, CPUBuilder, CPUTraverser, GPUTraverser
from .mesh_utils_impl import GPUMeshSampler, MeshSamplerMode


class GPURayTracer:
    def __init__(self, bvh_data):
        self.bvh_traverser = GPUTraverser(bvh_data)

    def trace(self, cam_poses, dirs, allow_negative=True, allow_backward=True, allow_forward=True):
        n_rays = cam_poses.shape[0]

        prim_idxs = torch.zeros((n_rays,),    dtype=torch.uint32,  device="cuda")
        mask      = torch.zeros((n_rays,),    dtype=torch.bool,    device="cuda")
        t         = torch.zeros((n_rays,),    dtype=torch.float32, device="cuda") + 1e9
        normals   = torch.zeros((n_rays, 3),  dtype=torch.float32, device="cuda")
        uvs       = torch.zeros((n_rays, 2),  dtype=torch.float32, device="cuda")

        self.bvh_traverser.ray_query(
            cam_poses,
            dirs,
            mask,
            t,
            prim_idxs,
            normals,
            uvs,
            allow_negative,
            allow_backward,
            allow_forward,
        )

        # t[t >= 1e9] = 0

        return mask, t, normals, uvs


class GPURayTracerAll:
    def __init__(self, bvh_data):
        self.bvh_traverser = GPUTraverser(bvh_data)

    def trace_all(self, cam_poses, dirs, max_hits_per_ray, allow_negative=False, allow_backward=True, allow_forward=True):
        n_rays = cam_poses.shape[0]

        prim_idxs = torch.zeros((n_rays, max_hits_per_ray), dtype=torch.uint32,  device="cuda")
        mask      = torch.zeros((n_rays, max_hits_per_ray), dtype=torch.bool,    device="cuda")
        t         = torch.zeros((n_rays, max_hits_per_ray), dtype=torch.float32, device="cuda") + 1e9
        n_hits    = torch.zeros((n_rays,),                  dtype=torch.uint32,  device="cuda")

        self.bvh_traverser.ray_query_all(
            cam_poses,
            dirs,
            mask,
            t,
            prim_idxs,
            n_hits,
            max_hits_per_ray,
            allow_negative,
            allow_backward,
            allow_forward,
        )

        # t[t >= 1e9] = 0

        return mask, t, prim_idxs.long(), n_hits.long()
