#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import mat2quat, quat2mat
from sklearn.mixture import GaussianMixture

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': 0, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz, scale, rotation, opacities, features, f_dc, f_rest = self.deparameterize_segments()

        print(f"Saving {xyz.shape[0]} points to {path}")

        xyz = xyz.data.cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = f_dc.data.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = f_rest.data.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacities.data.cpu().numpy()
        scale = scale.data.cpu().numpy()
        rotation = rotation.data.cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_prototypes(self, ply_path):

        plydata = PlyData.read(ply_path)

        # ------------------------------------------
        # Read big prototype gaussians into learnable tensors

        no_elements = plydata.elements[1]["x"].shape[0]

        xyz = np.stack((np.asarray(plydata.elements[1]["x"]),
                        np.asarray(plydata.elements[1]["y"]),
                        np.asarray(plydata.elements[1]["z"])),  axis=1)
        g = np.asarray(plydata.elements[1]["x"])
        
        opacities = np.asarray(np.zeros((no_elements, 1)))[..., np.newaxis]

        features_dc = np.zeros((no_elements, 3, 1))

        features_extra = np.zeros((no_elements, 3*(self.max_sh_degree + 1) ** 2 - 3))

        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[1].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((no_elements, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[1][attr_name])

        rot_names = [p.name for p in plydata.elements[1].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[1][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = (torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = (torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = (torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # ------------------------------------------
        # Read prototypes into non learnable tensors

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        segments = np.asarray(plydata.elements[0]["n_seg"])
        segments = segments.astype(np.int)

        self._proto_xyz = xyz
        self._proto_features_dc = features_dc#.transpose(1, 2).contiguous()
        self._proto_features_rest = features_extra#.transpose(1, 2).contiguous()
        self._proto_opacity = opacities
        self._proto_scaling = scales
        self._proto_rotation = rots
        self._proto_segments = segments

        self.active_sh_degree = self.max_sh_degree
    
    def parameterize_segments(self):
        ''' Parameterize segment gaussian with respect to the big gaussians'''

        '''print(f"means: {self._xyz.data[:2]}")
        print(f"rotations: {self._rotation.data[:2]}")
        print(f"scales: {self._scaling.data[:2]}")'''
        
        segment_Ms = self._xyz.data.cpu().numpy()
        segment_Rs = GaussianModel.get_matrix_from_quat(self._rotation.data).cpu().numpy()
        segment_Ss = np.exp(self._scaling.data.cpu().numpy())

        '''print(f"big pre means: {segment_Ms[:2]}")
        print(f"big pre rotations: {segment_Rs[:2]}")
        print(f"big pre scales: {segment_Ss[:2]}")'''

        self._original_segment_R = torch.tensor(segment_Rs, dtype=torch.float, device="cuda")

        param_dictionaries = [None] * segment_Ms.shape[0]

        for i in range(self._proto_xyz.shape[0]):

            n_seg = self._proto_segments[i]
            segment_M = segment_Ms[n_seg]
            segment_R = segment_Rs[n_seg]
            segment_S = segment_Ss[n_seg]
            proto_R = GaussianModel.get_matrix_from_quat(torch.tensor(self._proto_rotation[i], dtype=torch.float)).cpu().numpy()

            #print(f"{small_M.shape} {segment_R.shape} {segment_S.shape}")
            small_M = (self._proto_xyz[i] - segment_M) #@ segment_R #/ segment_S

            small_R = self._proto_rotation[i] #segment_R.T @ proto_R @ segment_R # 

            small_S = self._proto_scaling[i] # np.exp(self._proto_scaling[i]) / segment_S

            '''
            smallMeans_shifted = smallMeans - mean
            A_m = smallMeans_shifted @ R_big / S_big
            A_R = R_big.T @ smallR @ R_big 
            A_S = smallS / S_big
            '''

            if param_dictionaries[n_seg] is None:
                param_dictionaries[n_seg] = {
                    "xyz": np.array([small_M]),
                    "rotation": np.array([small_R]),
                    "scaling": np.array([small_S]),
                    "features_dc": np.array([self._proto_features_dc[i]]),
                    "features_rest": np.array([self._proto_features_rest[i]]),
                    "opacity": np.array([self._proto_opacity[i]])
                }
            else:
                param_dictionaries[n_seg]["xyz"] = np.concatenate((param_dictionaries[n_seg]["xyz"], np.array([small_M])), axis=0)
                param_dictionaries[n_seg]["rotation"] = np.concatenate((param_dictionaries[n_seg]["rotation"], np.array([small_R])), axis=0)
                param_dictionaries[n_seg]["scaling"] = np.concatenate((param_dictionaries[n_seg]["scaling"], np.array([small_S])), axis=0)
                param_dictionaries[n_seg]["features_dc"] = np.concatenate((param_dictionaries[n_seg]["features_dc"], np.array([self._proto_features_dc[i]])), axis=0)
                param_dictionaries[n_seg]["features_rest"] = np.concatenate((param_dictionaries[n_seg]["features_rest"], np.array([self._proto_features_rest[i]])), axis=0)
                param_dictionaries[n_seg]["opacity"] = np.concatenate((param_dictionaries[n_seg]["opacity"], np.array([self._proto_opacity[i]])), axis=0)

        self._param_dict = param_dictionaries

        # initilize _xyz to random noise
        self._xyz = nn.Parameter(0.05 * torch.randn_like(self._xyz).requires_grad_(True))

        '''print(f"param means: {self._param_dict[0]['xyz'][:2]}")
        print(f"param segment: {segment_Ms[0]}")
        print(f"param rotations: {small_Rs[:2]}")
        print(f"param scales: {small_Ss[:2]}")'''



    def deparameterize_segments(self, xyz):
        ''' Deparameterize segment gaussians with respect to the big gaussians
        return means, scales, quaternions'''
        
        segment_Ms = xyz
        segment_Rs = GaussianModel.get_matrix_from_quat(self._rotation.data)
        segment_Ss = torch.exp(self._scaling)  #np.exp(self._scaling.data.cpu().numpy())

        '''print(f"big post means: {segment_Ms[:2]}")
        print(f"big post rotations: {segment_Rs[:2]}")
        print(f"big post scales: {segment_Ss[:2]}")'''

        deparam_quats = []
        deparam_scales = []
        deparam_means = []
        deparam_features_dc = []
        deparam_features_rest = []
        deparam_opacities = []

        '''print(f"deparam begin means: {self._param_proto_xyz[:2]}")
        print(f"deparam begin rotations: {self._param_proto_rotation[:2]}")
        print(f"deparam begin scales: {self._param_proto_scaling[:2]}")'''

        for i in range(segment_Ms.shape[0]):
            small_xyz = torch.tensor(self._param_dict[i]["xyz"], dtype=torch.float, device="cuda")
            small_rot = torch.tensor(self._param_dict[i]["rotation"], dtype=torch.float, device="cuda")
            small_scale = torch.tensor(self._param_dict[i]["scaling"], dtype=torch.float, device="cuda")
            small_f_dc = torch.tensor(self._param_dict[i]["features_dc"], dtype=torch.float, device="cuda")
            small_f_rest = torch.tensor(self._param_dict[i]["features_rest"], dtype=torch.float, device="cuda")
            small_f_opacity = torch.tensor(self._param_dict[i]["opacity"], dtype=torch.float, device="cuda")

            #print(f"small_xyz: {small_xyz.shape}, small_rot: {small_rot.shape}, small_scale: {small_scale.shape}")

            segment_M = segment_Ms[i]
            segment_R = segment_Rs[i]
            segment_S = segment_Ss[i]

            small_M = small_xyz + segment_M # + (small_xyz * segment_S) #@ segment_R.T
            small_R = small_rot #self._original_segment_R[i] @ small_rot @ segment_R.T
            small_S = small_scale #* segment_S

            '''
            A_m_new = mean_shift + (A_m * S_scale) @ R_rot.T
            A_R_new = R_big @ A_R @ R_rot.T
            A_S_new = A_S * S_scale
            '''
            
            #rotation_quats = GaussianModel.quat_from_matrix(small_R)
            
            small_S_log = small_S #torch.log(small_S)


            deparam_quats.append(small_R) #rotation_quats)
            deparam_scales.append(small_S_log)
            deparam_means.append(small_M)
            deparam_features_dc.append(small_f_dc)
            deparam_features_rest.append(small_f_rest)
            deparam_opacities.append(small_f_opacity)


        #print(f"deparam_features_dc: {deparam_features_dc.shape}, deparam_features_rest: {deparam_features_rest.shape}, features_dx {self._features_dc.shape}, features_rest {self._features_rest.shape}")
        '''quats = deparam_quats #torch.tensor(deparam_quats, dtype=torch.float, device="cuda")
        scales = deparam_scales #torch.tensor(deparam_scales, dtype=torch.float, device="cuda")
        means = deparam_means #torch.tensor(deparam_means, dtype=torch.float, device="cuda")
        opacity = deparam_opacities #torch.tensor(deparam_opacities, dtype=torch.float, device="cuda")
        features_dc = deparam_features_dc.transpose(1, 2).contiguous() #torch.tensor(deparam_features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        features_rest = deparam_features_rest.transpose(1, 2).contiguous() #torch.tensor(deparam_features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        features = torch.cat((features_dc, features_rest), dim=1) '''

        quats = torch.cat(deparam_quats, axis=0)
        scales = torch.cat(deparam_scales, axis=0)
        means = torch.cat(deparam_means, axis=0)
        features_dc = torch.cat(deparam_features_dc, axis=0).transpose(1, 2).contiguous()
        features_rest = torch.cat(deparam_features_rest, axis=0).transpose(1, 2).contiguous()
        features = torch.cat((features_dc, features_rest), dim=1)
        opacity = torch.cat(deparam_opacities, axis=0)

        return means, scales, quats, opacity, features, features_dc, features_rest


    def get_matrix_from_quat(q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot_matrix = torch.empty((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
        rot_matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
        rot_matrix[..., 0, 1] = 2.0 * (xy - wz)
        rot_matrix[..., 0, 2] = 2.0 * (xz + wy)
        rot_matrix[..., 1, 0] = 2.0 * (xy + wz)
        rot_matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
        rot_matrix[..., 1, 2] = 2.0 * (yz - wx)
        rot_matrix[..., 2, 0] = 2.0 * (xz - wy)
        rot_matrix[..., 2, 1] = 2.0 * (yz + wx)
        rot_matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
        return rot_matrix

    def quat_from_matrix(rot_matrix):
        batch_size = rot_matrix.shape[0]
        quats = torch.empty((batch_size, 4), device=rot_matrix.device, dtype=rot_matrix.dtype)

        trace = rot_matrix[..., 0, 0] + rot_matrix[..., 1, 1] + rot_matrix[..., 2, 2]
        mask = trace > 0

        if mask.any():
            s = torch.sqrt(trace[mask] + 1.0) * 2
            quats[mask, 0] = 0.25 * s
            quats[mask, 1] = (rot_matrix[mask, 2, 1] - rot_matrix[mask, 1, 2]) / s
            quats[mask, 2] = (rot_matrix[mask, 0, 2] - rot_matrix[mask, 2, 0]) / s
            quats[mask, 3] = (rot_matrix[mask, 1, 0] - rot_matrix[mask, 0, 1]) / s

        return quats



    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
