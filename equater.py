import math
import torch
import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm


from modules.mesh_fusion.render import (
    features_to_world_space_mesh,
    render_mesh,
    edge_threshold_filter,
    unproject_points,
)
from utils.common_utils import (
    visualize_depth_numpy,
    save_rgbd,
)

import time
from utils.camera_utils import *

import utils.functions as functions
from utils.functions import rot_x_world_to_cam, rot_y_world_to_cam, rot_z_world_to_cam, colorize_single_channel_image, write_video
from modules.equilib import equi2pers, cube2equi, equi2cube

from modules.geo_predictors.PanoFusionDistancePredictor import PanoFusionDistancePredictor
from modules.inpainters import PanoPersFusionInpainter
from modules.geo_predictors import PanoJointPredictor
from modules.mesh_fusion.sup_info import SupInfoPool
from kornia.morphology import erosion, dilation
from scene.arguments import GSParams, CameraParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.graphics import focal2fov
from utils.loss import l1_loss, ssim

from SceneGraph import SceneGraph
from Segment import Segmentor
from modules.pose_sampler.circle_pose_sampler import CirclePoseSampler


'''
STRUCTURE

__init__
    initialize models and define shared variables
load_modules()
    在__init__函数中调用 加载两个模型inpainter, geo_predictor
project()
    mesh_to_perspective
    using render_mesh
    INPUT:world_to_perspective_camera_pose OUTPUT:rendered_image_tensor, rendered_image_pil
render_pano(self, pose):
    mesh_to_cubemap_to_panorama
    using project(), depth_to_distance(), cube2equi()
    INPUT:world_to_panorama_camera_pose OUTPUT:pano_rgb, pano_depth, pano_mask
rgbd_to_mesh()
    RGBD_to_mesh
    using features_to_world_space_mesh()
    INPUT:RGBD OUTPUT:None
    mesh iteration
find_depth_edge()
    depth_to_EdgeMask
    usingcv2.canny()
    INPUT:depth OUTPUT:EdgeMask
pano_distance_to_mesh()
    panoramaRGBD_to_mesh
    using rgbd_to_mesh
    INPUT:panoramaRGBD OUTPUT:None
    mesh iteration
load_inpaint_pose()
    create panorama_pose, pose
    using nothing
    INPUT:None OUTPUT:panorama_pose(NDArray), pose(list)
stage_inpaint_pano_greedy_search()
    选取完整度处于2/3分位的pose进行inpainting 完成mesh_iteration并且收集pseudo_view
    using render_pano(), inpaint_new_panorama(), geo_check(), pano_distance_to_mesh()
    INPUT:pose_dict OUTPUT:inpainted_panos_and_poses(list)
inpaint_new_panorama()
    inpainting
    using cv2.getStructuringElement(), inpainter.inpaint(), geo_predictor()
    INPUT:idx, RGBD, mask OUTPUT:inpainted_img, inpainted_distances, inpainted_normals
load_pano()
    加载panorama_init
    using resize_image_with_aspect_ratio(), pano_fusion_distance_predictor.predict()
    INPUT:Null OUTPUT:panorama_tensor, depth
load_camera_poses()
    create inpainted_poses_dict
    using nothing
    INPUT:None OUTPUT:pose_dict
pano_to_perpective()
    panorama_to_perspective
    using equi2pers()
    INPUT:panorama, pitch, yaw, fov OUTPUT:perspective
pano_to_cubemap()
    panorama_to_cubemap
    using pano_to_perspective()
    INPUT:panorama OUTPUT:cubemap, cubelap_depth
train_GS()
eval_GS()
find_object()
    find_object_in_vertices_tensor_given_segment_label
    using None
    INPUT:label OUTPUT:start, end
look_at()
    get_certain_camera_posse
    using None
    INPUT:obj, cam OUTPUT:R_44
load_object_poses()
    load_camera_a_series_ofposes_surrounding_certain_object
    using look_at()
    INPUT:obj_posi, main_cam_posi, obj_size OUTPUT:camera_poses_dict
object_inpaint()
    get_object_inpainted_poses_and_inpaint_and_iterate_mesh
    using stage_inpaint_pano_greedy_search()
    INPUT:label, world OUTPUT:object_inpainted_panos_and_poses
run()
    A COMPLETE PIPELINE
'''

@torch.no_grad()
class Pano2RoomPipeline(torch.nn.Module):
    def __init__(self, attempt_idx=""):
        '''initialize models and define shared variables'''

        super().__init__()

        # renderer setting
        self.blur_radius = 0
        self.faces_per_pixel = 8
        self.fov = 90
        self.R, self.T = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]), torch.Tensor([[0., 0., 0.]])
        self.pano_width, self.pano_height = 1024 * 2, 512 * 2
        self.H, self.W = 512, 512
        self.device = "cuda:0"

        # initialize
        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device) 
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)  
        self.vertices = torch.empty((3, 0), device=self.device, requires_grad=False)# gaussian_train_data
        self.colors = torch.empty((4, 0), device=self.device, requires_grad=False)# 前3行表示颜色，第四行表示标签
        self.labels = torch.empty((3, 0), device=self.device, requires_grad=False)#pseudo_rgb
        self.faces = torch.empty((3, 0), device=self.device, dtype=torch.long, requires_grad=False)# gaussian_train_data
        self.pix_to_face = None

        self.pose_scale = 0.6
        self.pano_center_offset = (-0.2,0.3)
        self.inpaint_frame_stride = 20
        self.poses = []

        # create exp dir
        self.setting = f""
        apply_timestamp = True
        if apply_timestamp:
            timestamp = str(int(time.time()))[-8:]
            self.setting += f"-{timestamp}"
        self.save_path = f'output/Pano2Room-results'
        self.save_details = False

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("makedir:", self.save_path)

        self.world_to_cam = torch.eye(4, dtype=torch.float32).to(self.device)
        self.cubemap_w2c_list = functions.get_cubemap_views_world_to_cam()

        self.load_modules()

        #init scene_graph
        print('load SceneGraph……')
        with open('input/text.txt', 'r') as f:
            text = f.read()
        self.scene_graph = SceneGraph(text, exist=True, exist_path='output/20250312193327')

        self.scene_depth_max = 4.0228885328450446

        # init segmentor
        self.class_names = self.scene_graph.extract_objects_names()
        self.segmentor = Segmentor(self.H, self.W, self.fov, self.class_names)

        self.prompts = []
        self.pidx = []
        self.size = []
        self.size_factor = 1.5

        # circle_sampler parameters
        self.pose_sampler = {
            'traverse_ratios': [0.2, 0.4, 0.6],
            'n_anchors_per_ratio': [8, 8, 8]
        }


    def load_modules(self):
        '''在__init__函数中调用 加载两个模型inpainter, geo_predictor'''
        self.inpainter = PanoPersFusionInpainter(save_path=self.save_path)
        self.geo_predictor = PanoJointPredictor(save_path=self.save_path)



    def project(self, world_to_cam):
        '''
        mesh_to_perspective
        using render_mesh
        INPUT:world_to_perspective_camera_pose OUTPUT:rendered_image_tensor, rendered_image_pil
        '''

        # project mesh into pose and render (rgb, depth, mask)
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf, self.mesh = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors[:3],# 只用前三个，因为前三个表示颜色，第四个表示标签
            H=self.H,
            W=self.W,
            fov_in_degrees=self.fov,
            RT=world_to_cam,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel
        )
        # mask rendered_image_tensor
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask
                 
        # stable diffusion models want the mask and image as PIL images
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        '''以下的三个变量暂时未被使用'''
        self.inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        self.inpaint_mask_restore = self.inpaint_mask
        self.inpaint_mask_pil_restore = self.inpaint_mask_pil

        return rendered_image_tensor[:3, ...], rendered_image_pil



    def render_pano(self, pose):
        '''
        mesh_to_cubemap_to_panorama
        using project(), depth_to_distance(), cube2equi()
        INPUT:world_to_panorama_camera_pose OUTPUT:pano_rgb, pano_depth, pano_mask
        '''

        cubemap_list = [] 
        for cubemap_pose in self.cubemap_w2c_list:# self.cubemap_w2c_list于__init__中定义，本质上是pano_to_cubemap的六个坐标转换矩阵形成的列表
            pose_tmp = pose.clone()
            pose_tmp = cubemap_pose.cuda().float() @ pose_tmp.float()# world_to_pano@pano_to_cubemap_sub_i=world_to_cubemap_sub_i 注意可能被名称误导
            rendered_image_tensor, rendered_image_pil = self.project(pose_tmp.cuda())# 渲染cubemap

            rgb_CHW = rendered_image_tensor.squeeze(0).cuda()
            depth_CHW = self.rendered_depth.unsqueeze(0).cuda()
            distance_CHW = functions.depth_to_distance(depth_CHW)
            mask_CHW = self.inpaint_mask.unsqueeze(0).cuda()
            cubemap_list += [torch.cat([rgb_CHW, distance_CHW, mask_CHW], axis=0)]

        torch.set_default_tensor_type('torch.FloatTensor')
        pano_rgbd = cube2equi(cubemap_list,
                                "list",
                                1024,2048)# CHW
        '''六个cubemap拼接形成pano 随后进行切片'''

        pano_rgb = pano_rgbd[:3,:,:]
        pano_depth =  pano_rgbd[3:4,:,:].squeeze(0)
        pano_mask =  pano_rgbd[4:,:,:].squeeze(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return pano_rgb, pano_depth, pano_mask# CHW, HW, HW



    def rgbd_to_mesh(self, rgbl, depth, world_to_cam=None, mask=None, pix_to_face=None, using_distance_map=False, pseudo=False):
        '''
        RGBD_to_mesh
        using features_to_world_space_mesh()
        INPUT:RGBD OUTPUT:None
        mesh iteration
        '''
        
        predicted_depth = depth.cuda()
        rgbl = rgbl.squeeze(0).cuda()

        if world_to_cam is None:
            world_to_cam = torch.eye(4, dtype=torch.float32)
        world_to_cam = world_to_cam.cuda()
        '''未提供位姿时 假设相机坐标系与世界坐标系对齐'''
        if pix_to_face is not None:
            self.pix_to_face = pix_to_face
        '''self.pix_to_face=None'''
        if mask is None:
            self.inpaint_mask = torch.ones_like(predicted_depth)
        else:
            self.inpaint_mask = mask
        '''未提供掩码时 处理全部像素'''

        if self.inpaint_mask.sum() == 0:
            return

        vertices, faces, colors = features_to_world_space_mesh(
                colors=rgbl,
                depth=predicted_depth,
                fov_in_degrees=self.fov,
                world_to_cam=world_to_cam,
                mask=self.inpaint_mask,
                pix_to_face=self.pix_to_face,
                faces=self.faces,
                vertices=self.vertices,
                using_distance_map=using_distance_map,
                edge_threshold=0.05
        )
        '''完成新mesh的生成'''

        # if not pseudo:

        faces += self.vertices.shape[1]# 面索引偏移 避免与现有顶点冲突

        self.vertices_restore = self.vertices.clone()
        self.colors_restore = self.colors.clone()
        self.faces_restore = self.faces.clone()
        '''保存 方便回退'''

        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)
        '''合并mesh'''

        # else:

        #     self.labels_restore = self.labels.clone()
        #     '''保存 方便回退'''

        #     self.labels = torch.cat([self.labels, colors_or_labels], dim=1)
        #     '''合并mesh'''



    def find_depth_edge(self, depth, dilate_iter=0):
        '''
        depth_to_EdgeMask
        usingcv2.canny()
        INPUT:depth OUTPUT:EdgeMask
        '''

        gray = (depth/depth.max() * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 60, 150)
        if dilate_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
        return edges



    def pano_distance_to_mesh(self, pano_rgbl, pano_distance, depth_edge_inpaint_mask, pose=None, pseudo=False):
        '''
        panoramaRGBD_to_mesh
        using rgbd_to_mesh
        INPUT:panoramaRGBD OUTPUT:None
        mesh iteration
        '''
        self.rgbd_to_mesh(pano_rgbl, pano_distance, mask=depth_edge_inpaint_mask, using_distance_map=True, world_to_cam=pose, pseudo=pseudo)
 


    def load_inpaint_poses(self):
        '''
        create inpainted_poses_dict
        using nothing
        INPUT:None OUTPUT:pose_dict
        '''

        pano_rgb, pano_distance, pano_mask = self.render_pano(self.world_to_cam)# 在初始位置下渲染一张panorama

        pose_dict = {}# {idx:pose, ...} # pose are c2w
        '''IMPORTANT!!!'''

        key = 0

        sampled_inpaint_poses = self.poses[::self.inpaint_frame_stride]# 在原始相机位姿字典中 每隔self.inpaint_frame_stride=20取样
        for anchor_idx in range(len(sampled_inpaint_poses)):
            pose = torch.eye(4).float()# pano pose dosen't support rotations\

            pose_44 = sampled_inpaint_poses[anchor_idx].clone()
            pose_44 = pose_44.float()
            Rw2c = pose_44[:3,:3].cpu().numpy()
            Tw2c = pose_44[:3,3:].cpu().numpy()
            yz_reverse = np.array([[1,0,0], [0,1,0], [0,0,1]])
            Rc2w = np.matmul(yz_reverse, Rw2c).T
            Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
            Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0) 
            pose[:3, 3] = torch.tensor(Pc2w[:3, 3]).cuda().float()
            pose[:3, 3] *= -1
            pose_dict[key] = pose.clone()

            key += 1
        return  pose_dict



    def stage_inpaint_pano_greedy_search(self, pose_dict):
        '''
        选取完整度处于2/3分位的pose进行inpainting 完成mesh_iteration并且收集pseudo_view
        using render_pano(), inpaint_new_panorama(), geo_check(), pano_distance_to_mesh()
        INPUT:pose_dict OUTPUT:inpainted_panos_and_poses(list)
        '''

        print("stage_inpaint_pano_greedy_search")
        pano_rgb, pano_distance, pano_mask = self.render_pano(self.world_to_cam)
        '''initialize原始pano 与修复后的版本作几何检查'''

        inpainted_panos_and_poses = []
        while len(pose_dict) > 0:
            print(f"len(pose_dict):{len(pose_dict)}")

            values_sampled_poses = []
            keys = list(pose_dict.keys())# 获取dict的key形成list
            for key in keys:# 遍历pose_dict
                pose = pose_dict[key]
                pano_rgb, pano_distance, pano_mask = self.render_pano(pose.cuda())# 渲染每一pose下的panorama
                view_completeness = torch.sum((1 - pano_mask * 1))/(pano_mask.shape[0] * pano_mask.shape[1])# 计算完整度
                
                values_sampled_poses += [(key, view_completeness, pose)]
                torch.cuda.empty_cache() 
            if len(values_sampled_poses) < 1:
                break

            # find inpainting with least view completeness
            values_sampled_poses = sorted(values_sampled_poses, key=lambda item: item[1])# 对完成度排序
            # least_complete_view = values_sampled_poses[0]
            least_complete_view = values_sampled_poses[len(values_sampled_poses)*2//3]# 筛选后2/3分位的视图位置

            key, view_completeness, pose = least_complete_view
            print(f"least_complete_view:{view_completeness}")
            del pose_dict[key]# 取出这一pose后从dict中删除
            '''选取完整度2/3分位的pose进行inpaint 处理既可能不完整但是不是最困难的pose'''

            # rendering rgb depth mask
            pano_rgb, pano_distance, pano_mask = self.render_pano(pose.cuda())# 渲染出此时视图

            # inpaint pano
            colors = pano_rgb.permute(1,2,0).clone()
            distances = pano_distance.unsqueeze(-1).clone()
            pano_inpaint_mask = pano_mask.clone()
            '''
            print("colors shape:", colors.shape)
            print("labels shape:", labels.shape)
            print("distances shape:", distances.shape)
            print("distances.squeeze(2) shape:", distances.squeeze(2).shape)
            '''
            '''inpainting过程'''

#            if pano_inpaint_mask.min().item() < .5:# 如果存在需要补全的部分
                # inpainting pano
            colors, distances, normals = self.inpaint_new_panorama(idx=key, colors=colors, distances=distances.squeeze(2), pano_mask=pano_inpaint_mask)# HWC, HWC, HW
            labels = torch.zeros((1, colors.shape[1]), dtype=torch.float)
            colors = torch.cat([colors, labels], dim = 0)
            '''inpainting过程'''

            # apply_GeoCheck:
            perf_pose = pose.clone().type(torch.float)
            perf_pose[0,3], perf_pose[1,3], perf_pose[2,3] = -pose[0,3], pose[2,3], 0 
            # self.pano_width=torch.tensor(self.pano_width,dtype=torch.float32)
            # self.pano_height=torch.tensor(self.pano_height,dtype=torch.float32)
            rays = gen_pano_rays(perf_pose.cuda(), self.pano_height, self.pano_width)
            # rays = gen_pano_rays(perf_pose.cuda(), self.pano_height.cuda(), self.pano_width.cuda())
            conflict_mask = self.sup_pool.geo_check(rays, distances.unsqueeze(-1))# 0 conflict, 1 not conflict
            pano_inpaint_mask = pano_inpaint_mask * conflict_mask 
            
            # add new mesh
            self.pano_distance_to_mesh(colors.permute(2,0,1), distances.squeeze(1), pano_inpaint_mask, pose=pose)# CHW, HW, HW
            # self.pano_distance_to_mesh(colors.permute(2,0,1), distances.squeeze(2), pano_inpaint_mask, pose=pose)# CHW, HW, HW
            # self.pano_distance_to_mesh(labels.permute(2,0,1), distances.squeeze(1), pano_inpaint_mask, pose=pose, pseudo=True)# CHW, HW, HW
            # self.pano_distance_to_mesh(labels.permute(2,0,1), distances.squeeze(2), pano_inpaint_mask, pose=pose, pseudo=True)# CHW, HW, HW

            # apply_GeoCheck:
            sup_mask = pano_inpaint_mask.clone()
            self.sup_pool.register_sup_info(pose=perf_pose.cuda(), mask=sup_mask.cuda(), rgb=colors.cuda(), distance=distances.unsqueeze(-1), normal=normals)
            
            # save renderred
            panorama_tensor_pil = functions.tensor_to_pil(pano_rgb.unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/renderred_pano_{key}.png")
            if self.save_details:
                depth_pil = Image.fromarray(colorize_single_channel_image(pano_distance.unsqueeze(0)/self.scene_depth_max))
                depth_pil.save(f"{self.save_path}/renderred_depth_{key}.png")        
                inpaint_mask_pil = Image.fromarray(pano_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                inpaint_mask_pil.save(f"{self.save_path}/mask_{key}.png")  
                inpaint_mask_pil = Image.fromarray(pano_inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                inpaint_mask_pil.save(f"{self.save_path}/inpaint_mask_{key}.png")  

            # save inpainted
            panorama_tensor_pil = functions.tensor_to_pil(colors.permute(2,0,1).unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/inpainted_pano_{key}.png")
            depth_pil = Image.fromarray(colorize_single_channel_image(distances.unsqueeze(0)/self.scene_depth_max))
            depth_pil.save(f"{self.save_path}/inpainted_depth_{key}.png") 

            # collect pano images for GS training
            inpainted_panos_and_poses += [(colors.permute(2,0,1).unsqueeze(0), pose.clone())] #BCHW, 44
            
        return inpainted_panos_and_poses



    def inpaint_new_panorama(self, idx, colors, distances, pano_mask):
        '''
        inpainting
        using cv2.getStructuringElement(), inpainter.inpaint(), geo_predictor()
        INPUT:idx, RGBD, mask OUTPUT:inpainted_img, inpainted_distances, inpainted_normals
        '''

        print(f"inpaint_new_panorama")

        # must dilate mask first
        mask = pano_mask.unsqueeze(-1)
        s_size = (9, 9)
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, s_size)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)
        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask.permute(0, 2, 3, 1).contiguous().squeeze(0).squeeze(-1)
        '''扩大需修复区域 确保边缘覆盖完整''' 

        distances = distances.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        inpainted_distances = None
        inpainted_normals = None

        inpainted_img = self.inpainter.inpaint(idx, colors, mask)

        # Keep renderred part
        inpainted_img = colors * (1 - mask) + inpainted_img * mask# 仅改变掩码部分
        inpainted_img = inpainted_img.cuda()

        inpainted_distances, inpainted_normals = self.geo_predictor(idx,
                                                                    inpainted_img,
                                                                    distances,
                                                                    mask=mask,
                                                                    reg_loss_weight=0.,
                                                                    normal_loss_weight=5e-2,
                                                                    normal_tv_loss_weight=5e-2)
        '''深度估计+法线预测'''
        inpainted_distances = inpainted_distances.squeeze()
        return inpainted_img, inpainted_distances, inpainted_normals



    def load_pano(self):
        '''
        加载panorama_init
        using resize_image_with_aspect_ratio(), pano_fusion_distance_predictor.predict()
        INPUT:Null OUTPUT:panorama_tensor, depth 
        '''

        image_path = f"input/pano.png"
        image = Image.open(image_path)
        if image.size[0] < image.size[1]:
            image = image.transpose(Image.TRANSPOSE)
        image = functions.resize_image_with_aspect_ratio(image, new_width=self.pano_width)
        panorama_tensor = torch.tensor(np.array(image))[...,:3].permute(2,0,1).float()/255
        # panorama_image_pil = functions.tensor_to_pil(panorama_tensor)

        depth_scale_factor = 3.4092

        # get panofusion_distance
        pano_fusion_distance_predictor = PanoFusionDistancePredictor()
        depth = pano_fusion_distance_predictor.predict(panorama_tensor.permute(1,2,0))# input:HW3
        depth = depth/depth.max() * depth_scale_factor
        print(f"pano_fusion_distance...[{depth.min(), depth.mean(),depth.max()}]")
        
        return panorama_tensor, depth# panorama_tensor:CHW, depth:HW
        # return panorama_tensor, None



    def load_camera_poses(self, pano_center_offset=[0,0]):# panorama_camera中心偏移量默认为0
        '''
        create panorama_pose, pose
        using nothing
        INPUT:None OUTPUT:panorama_pose(NDArray), pose(list)
        '''

        subset_path = f'input/Camera_Trajectory'# initial 6 poses are cubemaps poses
        files = os.listdir(subset_path)

 #       self.scene_depth_max = 4.0228885328450446

        pano_pose_44 = None
        pose_files = [f for f in files if f.startswith('camera_pose')]
        pose_files = sorted(pose_files)
        poses_name = pose_files
        poses = []
        for i, pose_name in enumerate(poses_name):
            with open(f'{subset_path}/{pose_name}', 'r') as f: 
                lines = f.readlines()
            pose_44 = []
            for line in lines:
                pose_44 += line.split()
            pose_44 = np.array(pose_44).reshape(4, 4).astype(float)
            if pano_pose_44 is None:
                pano_pose_44 = pose_44.copy()
                pano_pose_44_cubemaps = pose_44.copy()
                pano_pose_44[0,3] += pano_center_offset[0]
                pano_pose_44[2,3] += pano_center_offset[1]
            
            if i < 6:
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44_cubemaps)  
            else:
                ### convert gt_pose to gt_relative_pose with pano_pose
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44)
            '''对前六个cubemap 不作平移处理'''

            pose_relative_44 = np.vstack((-pose_relative_44[0:1,:], -pose_relative_44[1:2,:], pose_relative_44[2:3,:], pose_relative_44[3:4,:]))
            pose_relative_44 = pose_relative_44 @ rot_z_world_to_cam(180).cpu().numpy()

            pose_relative_44[:3,3] *= self.pose_scale
            poses += [torch.tensor(pose_relative_44).float()]# w2c
            '''relative:以第一个位姿pano_pose_44为基准 计算其他位姿的相对值 相当于形成了w2c'''

        return pano_pose_44, poses



    def pano_to_perpective(self, pano_bchw, pitch, yaw, fov):
        '''
        panorama_to_perspective
        using equi2pers()
        INPUT:panorama, pitch, yaw, fov OUTPUT:Perspective
        '''

        rots = {
            'roll': 0.,
            'pitch': pitch,# rotate vertical
            'yaw': yaw,# rotate horizontal
        }
        '''pitch:俯仰角ψ yaw:偏航角θ'''

        perspective = equi2pers(
            equi=pano_bchw.squeeze(0),
            rots=rots,
            height=self.H,
            width=self.W,
            fov_x=fov,
            mode="bilinear",
        ).unsqueeze(0)# BCHW

        return perspective



    def pano_to_cubemap(self, pano_tensor, pano_depth_tensor=None):# BCHW, HW
        '''
        panorama_to_cubemap
        using pano_to_perspective()
        INPUT:panorama OUTPUT:cubemap, cubelap_depth
        '''

        '''注意这里INPUT:pano_depth_tensor=None && OUTPUT:cubemaps_depth=None'''

        cubemaps_pitch_yaw = [(0, 0), (0, 3/2 * np.pi), (0, 1 * np.pi), (0, 1/2 * np.pi),\
                            (-1/2 * np.pi, 0), (1/2 * np.pi, 0)]
        pitch_yaw_list = cubemaps_pitch_yaw
        '''pitch:俯仰角ψ yaw:偏航角θ'''

        cubemaps = []
        cubemaps_depth = []
        # collect fov 90 cubemaps
        for view_idx, (pitch, yaw) in enumerate(pitch_yaw_list):
            view_rgb = self.pano_to_perpective(pano_tensor, pitch, yaw, 90)
            cubemaps += [view_rgb.cpu().clone()]
            if pano_depth_tensor is not None:
                view_depth = self.pano_to_perpective(pano_depth_tensor.unsqueeze(0).unsqueeze(0), pitch, yaw, 90)
                cubemaps_depth += [view_depth.cpu().clone()]
        return cubemaps, cubemaps_depth# BCHW, BCHW



    def train_GS(self):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        iterable_gauss = range(1, self.opt.iterations + 1)

        for iteration in iterable_gauss:
            self.gaussians.update_learning_rate(iteration)

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam, mesh_pose = viewpoint_stack[iteration%len(viewpoint_stack)]

            # Render GS
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            render_image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
            
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(render_image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))
            loss.backward()

            if self.save_details:
                if iteration % 200 == 0:
                    functions.write_image(f"{self.save_path}/Train_Ref_rgb_{iteration}.png", gt_image.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.)
                    functions.write_image(f"{self.save_path}/Train_GS_rgb_{iteration}.png", render_image.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.)

            with torch.no_grad():
                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)



    def eval_GS(self, eval_GS_cams):
        viewpoint_stack = eval_GS_cams
        l1_val = 0
        ssim_val = 0
        psnr_val = 0
        framelist = []
        depthlist = []
        for i in range(len(viewpoint_stack)):
            viewpoint_cam, mesh_pose = viewpoint_stack[i]

            results = render(viewpoint_cam, self.gaussians, self.opt, self.background) # 用这个方法来得到3DGS的渲染结果
            frame, depth = results['render'], results['depth'].detach().cpu()

            framelist.append(
                np.round(frame.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depthlist.append(colorize_single_channel_image(depth.detach().cpu()/self.scene_depth_max))

        if self.save_details:
            for i, frame in enumerate(framelist):
                image = Image.fromarray(frame, mode="RGB")
                image.save(os.path.join(self.save_path, f"Eval_render_rgb_{i}.png"))
                functions.write_image(f"{self.save_path}/Eval_render_depth_{i}.png", depthlist[i])
        
        write_video(f"{self.save_path}/GS_render_video.mp4", framelist[6:], fps=30)
        write_video(f"{self.save_path}/GS_depth_video.mp4", depthlist[6:], fps=30)
        print("Result saved at: ", self.save_path)
            


    def find_object(self, label):
        '''
        find_object_in_vertices_tensor_given_segment_label
        using None
        INPUT:label OUTPUT:object_tensor
        '''

        mask = ((self.colors[3] - label).abs < 1e-4)
        object_tensor = self.vertices[:, mask]
    
        return object_tensor

    def load_accompanied_poses(self, obj: torch.Tensor, main_cam_pos: torch.Tensor, size: float, pose_select_num=3, circle_num=2):
        '''
        load_camera_a_series_of_poses_surrounding_certain_object
        using look_at()
        INPUT:obj_posi, main_cam_pos, obj_size OUTPUT:camera_poses_dict
        '''

        '''
        print("obj shape:", obj.shape)
        print("main_cam shape:", main_cam.shape)
        '''
        
        obj_radius = torch.norm(obj - main_cam_pos) # 标量型tensor
        cam_radius_diff = obj_radius / circle_num
        '''这里的三个参数应该都与size有关 这里暂时处理成定值'''

        object_poses_dict = {}
        key = 0

        for i in range(circle_num):
            cam_radius = cam_radius_diff * (i + 1)
            r1 = obj_radius
            r2 = cam_radius
            center = (r2**2) / (2 * r1**2) * obj + (1 - (r2**2) / (2 * r1**2)) * main_cam_pos
            rho = r2 / (2 * r1) * math.sqrt(4 * r1**2 - r2**2)
            z = (obj - main_cam_pos)/r1
            up = torch.tensor([0, 1, 1])
            x:torch.Tensor = torch.cross(up, z) / torch.linalg.norm(torch.cross(up, z))
            y:torch.Tensor = torch.cross(z, x) / torch.linalg.norm(torch.cross(z, x))

            theta0 = 2 * torch.pi / pose_select_num / circle_num * i
            
            for j in range(pose_select_num):

                theta = j / pose_select_num * 2 * torch.pi + theta0
                cam = center + rho * math.cos(theta) * x + rho * math.sin(theta) * y
                R = look_at(to_vec = (obj - cam).unsqueeze(0)).squeeze().T # 33
                T = - R @ cam.unsqueeze(1) # 31
                object_poses_dict[key] = torch.cat([R,T], dim = -1)
                key += 1

        return object_poses_dict


    def find_main_position(self, obj_center, main_select_position, size):
        
        pose_select_num = 3
        circle_num = 1
        poses = self.load_accompanied_poses(obj_center, main_select_position, size, pose_select_num, circle_num)
        select_position = {}
        for idx, pose in poses.items():
            _, _, pano_mask = self.render_pano(pose)
            view_completeness = torch.sum((1 - pano_mask * 1))/(pano_mask.shape[0] * pano_mask.shape[1])
            select_position[idx] = view_completeness
        sorted_selected_position = sorted(select_position.items(), key=lambda item: item[1], reverse=True)
        return poses[sorted_selected_position[0][0]]
        
        
    def pano_segment(self, pano_tensor):
        '''
        Args:
        - pano_tensor: tensor[3,H,W]
        - class_names: List[N]

        Returns:
        - label_tensor: tensor[H,W](float)
        - label_num
        - evironment_label
        '''
        
        # import Segment
        
        environment_label = self.segmentor.class_cnt
        label_tensor = self.segmentor.segment_pano(pano_tensor).unsqueeze(0)
        label_num = self.segmentor.tot

        return label_tensor, label_num, environment_label


    def run(self):
        
        # shutup.please()

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        pano_path = "input/pano.png"
        image = Image.open(pano_path)
        pano:torch.tensor = torch.tensor(np.array(image))[...,:3].permute(2,0,1).float()/255
        panorama_label, label_num, environment_label = self.pano_segment(pano)

        # Load Initial RGB, Depth, Label Tensor        
        panorama_rgb, panorama_depth = self.load_pano() # [C, H, W]
        panorama_rgb, panorama_depth = panorama_rgb.cuda(), panorama_depth.cuda() # [C, H, W], [H, W]


        # Load Initial Depth_Edge & Depth_Edge_Inpainted_Mask
        depth_edge = self.find_depth_edge(panorama_depth.cpu().detach().numpy(), dilate_iter=1)
        depth_edge_inpaint_mask = ~(torch.from_numpy(depth_edge).cuda().bool())# 反转edge_mask：inpainting过程中边缘不动
        # Save Depth_Edge as Image
        depth_edge_pil = Image.fromarray(depth_edge)# edge_mask格式->img
        depth_pil = Image.fromarray(visualize_depth_numpy(panorama_depth.cpu().detach().numpy())[0].astype(np.uint8))# depth格式->img
        _, _ = save_rgbd(depth_pil, depth_edge_pil, f'depth_edge', 0, self.save_path)


        # Registration
        self.sup_pool = SupInfoPool()
        self.sup_pool.register_sup_info(pose=torch.eye(4).cuda(),
                                        mask=torch.ones([self.pano_height, self.pano_width]),
                                        rgb=panorama_rgb.permute(1,2,0),
                                        distance=panorama_depth.unsqueeze(-1))
        self.sup_pool.gen_occ_grid(256)


        # Pano2Mesh
        panorama_rgbl = torch.cat([panorama_rgb, panorama_label], dim = 0) # [C+L, H, W]
        self.pano_distance_to_mesh(panorama_rgbl, panorama_depth, depth_edge_inpaint_mask)

        # Objects Inpainting
        camera_positions = CirclePoseSampler(panorama_depth, **self.pose_sampler) # [N, 3]
        object_centers = {}
        inpainted_panos_and_poses = []
        #  self.poses = []
        for label in range(label_num):
            if self.segmentor.id2type[label] == environment_label:
                continue
            object_tensor = self.find_object(label).T # [N, 3]
            size = self.size_factor * object_tensor.std(dim = 0).norm()
            self.size.append[size]
            object_centers[label] = object_tensor.mean(dim = 0) # dict(idx: [3,])
            cam_to_obj_vec: torch.Tensor = object_centers[label] - camera_positions # [N, 3]
            cam_to_obj_dis = cam_to_obj_vec.norm(dim = 1) # [n,]
            mask = cam_to_obj_dis > size
            if mask.any():
                idx = torch.argmin(cam_to_obj_dis[mask])
                idx = torch.nonzero(mask)[idx][0]
                main_select_position = camera_positions[idx]
                main_position = self.find_main_position(obj_center = object_centers[label], main_select_position = main_select_position, size = size)
                pose_dict = self.load_accompanied_poses(obj = object_centers[label], main_cam_pos = main_position, size = size)
                inpainted_panos_and_poses.extend(self.stage_inpaint_pano_greedy_search(pose_dict))
                # self.poses.extend(list(pose_dict.values()))


        # Global Completion
        R = look_at(-camera_positions) # c2w N33
        T = -(camera_positions @ R).unsqueeze(1) # T N31
        poses = torch.cat([R.permute(0,2,1), T], dim = -1)
        pose_dict = {}
        key = 0
        for pose in poses:
            pose_dict[key] = pose
            key = key + 1
        inpainted_panos_and_poses.extend(self.stage_inpaint_pano_greedy_search(pose_dict))
        # self.poses.extend(list(pose_dict.values()))


        # Train 3DGS
  #     self.poses = list(self.pose.values())
        self.opt = GSParams()# gaussian_train_parameters in ./scene/arguments.py
        self.cam = CameraParams()# camera_parameters in ./scene/arguments.py
        self.gaussians = GaussianModel(self.opt.sh_degree)# gaissian_ball_parameters in ./scene/gaussian_model.py
        self.opt.white_background = True
        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]# 背景颜色 但GSParams()中的self.white_background = False 此处可能是某种调整
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')# 填充背景色
        
        traindata = {
            'camera_angle_x': self.cam.fov[0],# in CameraParams(): self.fov = ((angle/180.0)*np.pi, (angle/180.0)*np.pi) 视场角宽度
            'W': self.W,
            'H': self.H,
            'pcd_points': self.vertices.detach().cpu(),#初始为空的顶点张量列表 形状3*0
            'pcd_colors': self.colors[:3].permute(1,0).detach().cpu(),#初始为空的顶点颜色列表 形状3*0
            'frames': [],#初始为空的frame_list 用于存放训练用视图
        }

        for inpainted_pano_images, pano_pose_44 in inpainted_panos_and_poses:

            cubemaps, cubemaps_depth = self.pano_to_cubemap(inpainted_pano_images) # BCHW
            for i in range(len(cubemaps)):
                inpainted_img = cubemaps[i]

                mesh_pose = self.cubemap_w2c_list[i].cuda() @ pano_pose_44.type(torch.float32).clone().cuda()

                pose_44 = mesh_pose.clone()
                pose_44 = pose_44.float()
                pose_44[0:1,:] *= -1
                pose_44[1:2,:] *= -1

                Rw2c = pose_44[:3,:3].cpu().numpy()
                Tw2c = pose_44[:3,3:].cpu().numpy()
                yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

                Rc2w = np.matmul(yz_reverse, Rw2c).T
                Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
                Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)  

                traindata['frames'].append({
                    'image': functions.tensor_to_pil(inpainted_img),
                    'transform_matrix': Pc2w.tolist(), 
                    'fovx': focal2fov(256, inpainted_img.shape[-1]),# focal2fov(focal, pixels) = 2*math.atan(pixels/(2*focal))
                    'mesh_pose': mesh_pose
                })
                print(i)
                '''frame的结构'''

        self.scene = Scene(traindata, self.gaussians, self.opt)   
        self.train_GS()
        outfile = self.gaussians.save_ply(os.path.join(self.save_path, '3DGS.ply'))


        # Eval GS
        evaldata = {
            'camera_angle_x': self.cam.fov[0],
            'W': self.W,
            'H': self.H,
            'frames': [],
        }
        
        self.poses = self.load_camera_poses(self.pano_center_offset)
        for i in range(len(self.poses)):
            gt_img = inpainted_img

            pose_44 = self.poses[i].clone()
            pose_44 = pose_44.float()
            pose_44[0:1,:] *= -1
            pose_44[1:2,:] *= -1

            Rw2c = pose_44[:3,:3].cpu().numpy()
            Tw2c = pose_44[:3,3:].cpu().numpy()
            yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

            Rc2w = np.matmul(yz_reverse, Rw2c).T
            Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
            Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)                  

            evaldata['frames'].append({
                'image': functions.tensor_to_pil(gt_img),
                'transform_matrix': Pc2w.tolist(), 
                'fovx': focal2fov(256, gt_img.shape[-1]),
                'mesh_pose': self.poses[i].clone()
            })

        from scene.dataset_readers import loadCamerasFromData
        eval_GS_cams = loadCamerasFromData(evaldata, self.opt.white_background)
        self.eval_GS(eval_GS_cams)





pipeline = Pano2RoomPipeline()
pipeline.run()