import torch
from .mesh_utils_impl import Mesh, BVHData, CPUBuilder, CPUTraverser, GPUTraverser
from .mesh_utils_impl import GPUMeshSampler, MeshSamplerMode


class GPURayTracer:
    def __init__(self, bvh_data):
        self.bvh_traverser = GPUTraverser(bvh_data)
        self.n_reserved = 100

    def reserve_arrays(self):
        self.depths    = torch.zeros((self.n_reserved,),    dtype=torch.int,     device="cuda")
        self.prim_idxs = torch.zeros((self.n_reserved,),    dtype=torch.uint32,  device="cuda")
        self.mask      = torch.zeros((self.n_reserved,),    dtype=torch.bool,    device="cuda")
        self.t         = torch.zeros((self.n_reserved,),    dtype=torch.float32, device="cuda") + 1e9
        self.normals   = torch.zeros((self.n_reserved, 3),  dtype=torch.float32, device="cuda")

    def trace(self, cam_poses, dirs):
        if cam_poses.shape[0] > self.n_reserved:
            self.n_reserved = cam_poses.shape[0]
            self.reserve_arrays()

        self.bvh_traverser.ray_query(
            cam_poses,
            dirs,
            self.mask,
            self.t,
            self.prim_idxs,
            self.normals,
        )

        return self.mask, self.t, self.normals
