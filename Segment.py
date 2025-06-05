# import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import build_sam, SamPredictor 

import argparse
import json
import os
import numpy as np
from typing import Any, Dict, List

from tqdm import tqdm
import colorsys
import random
import clip
import torch
from clip_text import class_names_coco, class_names_ADE_150
from PIL import Image 
import torch.nn.functional as F
from utils.camera_utils import img_to_pano_coord, pano_to_img_coord
from utils.camera_utils import look_at
from trimesh.creation import icosphere
import math
from utils import functions

def generate_spherical_cam_poses(subdivisions = 2) -> torch.Tensor:
    """
    生成20个相机以覆盖球面
    生成以原点为中心，覆盖球面的相机矩阵列表
    :param subdivisions: 20面体细分级别，控制采样密度
    :return: 形状为 [n, 3, 4] 的位姿矩阵，其中平移部分为原点，旋转部分覆盖球面
    """

    mesh = icosphere(subdivisions = subdivisions)
    vertices = torch.tensor(mesh.vertices, dtype = torch.float32)
    faces = torch.tensor(mesh.faces, dtype = torch.int32)
    to_vec = []
    for face in faces:
        to_vec.append(torch.mean(vertices[face, :], dim = 0, keepdim = True))

    vertices = torch.cat(to_vec, dim = 0)
    vertices = vertices / torch.norm(vertices, dim = 1, keepdim = True)

    n = vertices.shape[0]
    up_vec = torch.zeros(n, 3, dtype = torch.float32)
    z_mask = (vertices[:, 2].abs() > 0.99)
    up_vec[z_mask, 1] = 1.0
    up_vec[~z_mask, 2] = 1.0

    rot_mats = look_at(vertices, up_vec)

    poses = torch.cat([rot_mats,torch.zeros(n,3,1)], dim = -1)

    return poses

def equi2pers(pano, pose, fovx, H, W):
    '''
    全景图转化为透视图
    :param pano: 全景图，形状为 (pano_h, pano_w, 3) 的 numpy 数组
    :param pose: 相机的外参，形状为 (3, 4) 的 numpy 数组
    :param H: 透视图的高度
    :param W: 透视图的宽度
    :return: 透视图，形状为 (H, W, 3) 的 numpy 数组
    '''

    fovx_rad = math.radians(fovx)

    fx = W * 0.5 / math.tan(fovx_rad/2)
    fy = fx
    cx, cy = W / 2, H / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 提取外参中的旋转矩阵 R 和平移向量 T
    R = pose[:, :3]
    T = pose[:, 3:]

    # 生成透视图的像素坐标网格
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    pixel_coords = np.stack([x, y, np.ones_like(x)], axis=-1)  # 形状为 (H, W, 3)
    print("==========pixel_coords shape:", pixel_coords.shape, "==========")

    # 将像素坐标转换为归一化设备坐标（normalized device coordinates）
    normalized_coords = np.linalg.inv(K) @ pixel_coords.reshape(-1, 3).T  # 形状为 (3, H*W)
    print("==========normalized_coords:", normalized_coords.shape, "==========")

    # 将归一化设备坐标转换为世界坐标
    world_coords = R.T @ (normalized_coords - T)

    # 将世界坐标转换为全景图的球面坐标
    # z:forward, x:right, y:down
    theta = np.arctan2(world_coords[0], world_coords[2])  # 经度，范围 [-pi, pi]
    phi = np.arcsin(world_coords[1] / np.linalg.norm(world_coords[:3], axis=0))  # 纬度，范围 [-pi/2, pi/2]

    # 将球面坐标转换为全景图的像素坐标
    pano_x = (theta / (2 * np.pi) + 0.5) * pano.shape[1]  # 经度映射到全景图的宽度
    pano_y = (phi / np.pi + 0.5) * pano.shape[0]  # 纬度映射到全景图的高度
    print("==========pano_x:", pano_x.shape, "==========")
    print("==========pano_y:", pano_y.shape, "==========")

    # 使用双线性插值获取全景图中的像素值
    pano_x = np.clip(pano_x, 0, pano.shape[1] - 1)
    pano_y = np.clip(pano_y, 0, pano.shape[0] - 1)
    pano_x_floor = np.floor(pano_x).astype(np.int32)
    pano_x_ceil = np.ceil(pano_x).astype(np.int32)
    pano_y_floor = np.floor(pano_y).astype(np.int32)
    pano_y_ceil = np.ceil(pano_y).astype(np.int32)
    print("==========pano_x:", pano_x.shape, "==========")
    print("==========pano_y:", pano_y.shape, "==========")
    print("==========pano_x_floor:", pano_x_floor.shape, "==========")
    print("==========pano_x_ceil:", pano_x_ceil.shape, "==========")
    print("==========pano_y_floor:", pano_y_floor.shape, "==========")
    print("==========pano_y_ceil:", pano_y_ceil.shape, "==========")

    # 双线性插值
    dx = pano_x - pano_x_floor
    dy = pano_y - pano_y_floor
    pano_x_floor = np.clip(pano_x_floor, 0, pano.shape[1] - 1)
    pano_x_ceil = np.clip(pano_x_ceil, 0, pano.shape[1] - 1)
    pano_y_floor = np.clip(pano_y_floor, 0, pano.shape[0] - 1)
    pano_y_ceil = np.clip(pano_y_ceil, 0, pano.shape[0] - 1)
    print("==========dx:", dx.shape, "==========")
    print("==========dy:", dy.shape, "==========")
    print("==========pano_x_floor:", pano_x_floor.shape, "==========")
    print("==========pano_x_ceil:", pano_x_ceil.shape, "==========")
    print("==========pano_y_floor:", pano_y_floor.shape, "==========")
    print("==========pano_y_ceil:", pano_y_ceil.shape, "==========")

    # 获取四个邻近像素的值
    q11 = pano[pano_y_floor, pano_x_floor]
    q12 = pano[pano_y_ceil, pano_x_floor]
    q21 = pano[pano_y_floor, pano_x_ceil]
    q22 = pano[pano_y_ceil, pano_x_ceil]
    print("==========q11:", q11.shape, "==========")
    print("==========q12:", q12.shape, "==========")
    print("==========q21:", q21.shape, "==========")
    print("==========q22:", q22.shape, "==========")

    # 双线性插值计算
    b = dy[..., None]
    a = dx[..., None]
    
    assert (a.shape == (H*W,1))

    pers_img = (1 - a) * (1 - b) * q11 + a * (1 - b) * q21 + (1 - a) * b * q12 + a * b * q22
    print("==========pers_img:", pers_img.shape, "==========")
    
    # 将结果重塑为透视图的形状
    pers_img = pers_img.reshape(H, W, 3).astype(np.uint8)

    return pers_img


def pers2equi(poses, fov, H, W, images, pano_h, pano_w):
    """
    透视标签图拼接成全景图
    :param poses: np.ndarray [b,3,4]
    :param H: int
    :param W: int
    :param images: List[Tensor[H,W,3]]
    :param class_num: int
    :return: tensor [pano_h,pano_w]
    """

    fov_rad = math.radians(fov)

    fx = W*0.5/math.tan(fov_rad/2)
    fy = fx
    cx ,cy = W/2, H/2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype = np.float32)

    out = - torch.zeros((pano_h, pano_w), dtype=np.int32)

    px, py = np.meshgrid(np.arange(pano_w), np.arange(pano_h))
    theta, phi = px / pano_w * 2 * np.pi - np.pi, py / pano_h * np.pi - np.pi / 2

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    # z:forward, x:right, y:down

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    word_coord = np.stack((x, y, z), axis=0)

    for pose, image in zip(poses,images):
        R = pose[:,:3]
        T = pose[:,3:]

        cam_coord = R @ word_coord + T
        pixel_coord = K @ cam_coord

        pixel_coord[2][np.where(np.abs(pixel_coord[2]) < 1e-6)] = -1
        valid_index = np.where(np.logical_and.reduce((pixel_coord[2] > 0,
                                                      pixel_coord[0]/pixel_coord[2] >= 0,
                                                      pixel_coord[0]/pixel_coord[2] <= W - 1,
                                                      pixel_coord[1]/pixel_coord[2] >= 0,
                                                      pixel_coord[1]/pixel_coord[2] <= H - 1)))[0]
        pixel_coord = pixel_coord[:2, valid_index]/pixel_coord[-1:, valid_index] # 2N
        pixel_coord = np.round(pixel_coord).astype(np.int32)

        valid_index_pano_2d = np.unravel_index(valid_index, (pano_h,pano_w))

        out[valid_index_pano_2d[0], valid_index_pano_2d[1]] = image[pixel_coord[1], pixel_coord[0]]

    return out


def generate_text_embeddings(classnames, templates, model, device):
    # 生成类列表的词嵌入
    with torch.no_grad():
        class_embeddings_list = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embedding = model.encode_text(texts) #embed with text encoder
            class_embeddings_list.append(class_embedding)
        class_embeddings = torch.stack(class_embeddings_list, dim=1).to(device)
    return class_embeddings


def segment(images:List, class_names:List):
    '''
    像素标注，返回各像素的分类索引矩阵
    images: RGB image list[ndarray[H,W,3]]
    class_names: list[n]
    
    return: List[Tensor[h,w,n]]
    '''
    device = torch.device("cuda")
    results = []

    # 加载SAM
    checkpoint = './checkpoint/sam_vit_h_4b8939.pth'
    mode_type = 'vit_h'
    sam = sam_model_registry[mode_type](checkpoint = checkpoint)
    masks_generator = SamAutomaticMaskGenerator(sam)

    # 加载CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_features = generate_text_embeddings(class_names, ['a clean origami {}.'], clip_model, device)#['a rendering of a weird {}.'], model)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.squeeze(0)

    for image in tqdm(images):
        anns = masks_generator.generate(image)
        result = -1 * torch.ones((image.shape[0],image.shape[1]), dtype = torch.int32)
        for i, ann in enumerate(anns):
            mask = ann['segmentation']
            image_new = image.copy()
            ind = np.where(mask)
            image_new[mask == 0] = 0
            y1, x1, y2, x2 = min(ind[0]), min(ind[1]), max(ind[0]), max(ind[1])
            image_new = Image.fromarray(image_new[y1:y2+1, x1:x2+1])
            image_new = preprocess(image_new)

            image_features = clip_model.encode_image(image_new.unsqueeze(0).to(device))
            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features.float() @ text_features.float().T).softmax(dim=-1)
            # values, indices = similarity[0].topk(1)
            index = torch.argmax(similarity[0], dim = -1)
            if similarity[0][index] > 0.8:
                result[ind[0], ind[1]] = int(index)

        results.append(result)

    return results


def apply_mask(image, mask):
    # 叠加mask于原图
    image_new = image.copy()
    color_mask = (np.random.random(3)*255).astype(np.uint8)
    img = np.zeros((mask.shape[0],mask.shape[1],3),dtype = np.uint8)
    img[mask] = color_mask
    image_new = image_new * (1-0.35) + img * 0.35
    return Image.fromarray(image_new.astype(np.uint8))
    # return image_new


def visualize(image, class_num):
    # 分类结果可视化
    colors = np.linspace(0, 255, class_num + 1).astype(np.uint8)
    image[:, :] = colors[image[:, :]]
    return Image.fromarray(image.astype(np.uint8), mode = 'L')

def segment_pano(pano, class_names):
    '''
    :pano: ndarray[H,W,3]
    :class_names: List[n]
    :return: tensor[H,W]
    '''

    print('加载相机……')
    poses = functions.get_cubemap_views_world_to_cam()
    poses = [pose[:3].cpu().numpy() for pose in poses]
    fov = 90
    H, W = 512, 512

    images = []
    print('加载透视图……')
    for i, pose in enumerate(poses):
        images.append(equi2pers(pano, pose, fov, 512, 512))
        Image.fromarray(images[i]).save(f'output/20250312193327/seg/pers/{i}.jpg')

    print('分割透视图……')
    segmented_images = segment(images, class_names)

    for i, segmented_image in enumerate(segmented_images):
        visualize(segmented_image.detach().cpu().numpy(), len(class_names)).save(
            f'output/20250312193327/seg/segpers/{i}.jpg')

    print('映射回全景图……')
    out = pers2equi(poses, fov, H, W, segmented_images, pano.shape[0], pano.shape[1]) # tensor

    print('分割结果可视化……')
    segmented_pano = visualize(out, len(class_names))
    # segmented_pano.show()
    segmented_pano.save('output/20250312193327/seg_pano.jpg')

    return out

if __name__ == "__main__":

    from SceneGraph import SceneGraph
    scenegraph = SceneGraph(exist = True, exist_path = "output/20250312193327")
    class_names = scenegraph.extract_objects_names()
    
    pano_path = ""
    pano_pil = Image.open(pano_path)
    pano:torch.tensor = torch.from_numpy(np.array(pano_pil))#HW3

    out = segment_pano(pano, class_names)