import cv2
import numpy as np
import lpips
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda()

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(img1_path, img2_path):
    # 加载图像
    transform = transforms.Compose([
     transforms.ToTensor()  # 将图像转换为张量，范围 [0, 1]
    ])

    # 打开图像文件
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # 将图像转换为张量
    img1_tensor = transform(img1).unsqueeze(0)  # 增加批量维度
    img2_tensor = transform(img2).unsqueeze(0)

    # 确保图像大小相同
    if img1_tensor.shape != img2_tensor.shape:
        raise ValueError("输入图像大小必须相同")

    # 计算 SSIM 值
    window_size = 11
    size_average = True

    ssim_value = ssim(img1_tensor, img2_tensor, window_size=window_size, size_average=size_average)

    # print("SSIM 值为:", ssim_value.item())
    return ssim_value.item()

def calculate_psnr(img1_path, img2_path):
    # 读取原始图像和重建图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)

    # 计算PSNR
    max_pixel_value = 255  # 对于8位图像，最大像素值为255
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    # print(f"PSNR: {psnr} dB")
    return psnr

lpips_model = lpips.LPIPS(net="alex")

def calculate_lpips(img1_path, img2_path):
    # 加载图像文件
    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)

    # 加载预训练的LPIPS模型

    # 将图像转换为PyTorch的Tensor格式
    image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 使用LPIPS模型计算距离
    distance = lpips_model(image1_tensor, image2_tensor)

    # print("LPIPS distance:", distance.item())
    return distance.item()

import os
path_GS = "Pano2Room-results630/renderred_GS"
path_mesh = "Pano2Room-results701"
PSNR = []
SSIM = []
LPIPS = []

for i in range(141):
    img1_path = os.path.join(path_GS, f"{i}.png")
    img2_path = os.path.join(path_mesh, F'Eval_render_rgb_{i}.png')
    LPIPS.append(calculate_lpips(img1_path, img2_path))
    PSNR.append(calculate_psnr(img1_path, img2_path))
    SSIM.append(calculate_ssim(img1_path, img2_path))

with open("metrics.txt", "w") as f:
    f.write(f"PSNR:  {sum(PSNR)/len(PSNR)}\n")
    f.write(f"SSIM:  {sum(SSIM)/len(SSIM)}\n")
    f.write(f"LPIPS:  {sum(LPIPS)/len(LPIPS)}\n")