import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .warp_utils import flow_warp


class ReBlur(nn.Module):
    def __init__(self):
        super(ReBlur, self).__init__()

    def forward(self, blur, sharp, kernel, weight):
        """
        S0-S1,S0-S2,S0-S3,S0-S4,S0-Sn
        """
        _, N, _, _ = weight.shape  # N条光流
        re_blur = flow_warp(sharp, torch.cat([kernel[:, 0:1, :, :], kernel[:, N:N + 1, :, :]], dim=1)) * weight[:, 0:1, :, :]  # 用第0条光流warp出第0帧
        for i in range(1, N):
            flow = torch.cat([kernel[:, i:i + 1, :, :], kernel[:, i + N:i + N + 1, :, :]], dim=1)  # 第i条光流
            frame = flow_warp(sharp, flow)  # 第i帧
            frame = frame * weight[:, i:i + 1, :, :]  # 第i帧乘以权重
            re_blur = re_blur + frame  # N个帧累加
        return F.mse_loss(re_blur, blur)


class ReBlurLinear(nn.Module):
    def __init__(self):
        super(ReBlurLinear, self).__init__()

    def forward(self, blur, sharp, kernel, weight):
        """
        S0-S1-S2-S3-S4-Sn
        """
        _, N, _, _ = weight.shape
        frame = flow_warp(sharp, torch.cat([kernel[:, 0:1, :, :], kernel[:, N:N + 1, :, :]], dim=1))  # 根据sharp图像warp出第一帧
        re_blur = frame * weight[:, 0:1, :, :]  # warp出的帧，乘以自己对应的权重
        for i in range(1, N):
            flow = torch.cat([kernel[:, i:i + 1, :, :], kernel[:, i + N:i + N + 1, :, :]], dim=1)  # 第i条光流
            frame = flow_warp(frame, flow)  # 对前一帧warp，得到下一帧，同时这个下一帧，也是下次循环的前一帧
            re_blur = re_blur + frame * weight[:, i:i + 1, :, :]  # 第i帧乘以权重，并累加
        return F.mse_loss(re_blur, blur)


class ReBlurL1(nn.Module):
    def __init__(self):
        super(ReBlurL1, self).__init__()

    def forward(self, blur, sharp, kernel, weight):
        """
        S0-S1,S0-S2,S0-S3,S0-S4,S0-Sn
        """
        _, N, _, _ = weight.shape  # N条光流
        re_blur = flow_warp(sharp, torch.cat([kernel[:, 0:1, :, :], kernel[:, N:N + 1, :, :]], dim=1)) * weight[:, 0:1, :, :]  # 用第0条光流warp出第0帧
        for i in range(1, N):
            flow = torch.cat([kernel[:, i:i + 1, :, :], kernel[:, i + N:i + N + 1, :, :]], dim=1)  # 第i条光流
            frame = flow_warp(sharp, flow)  # 第i帧
            frame = frame * weight[:, i:i + 1, :, :]  # 第i帧乘以权重
            re_blur = re_blur + frame  # N个帧累加
        return F.l1_loss(re_blur, blur)


class ReBlurImage(nn.Module):
    def __init__(self):
        super(ReBlurImage, self).__init__()

    def forward(self, blur, sharp, kernel, weight):
        """
        S0-S1,S0-S2,S0-S3,S0-S4,S0-Sn
        """
        _, N, _, _ = weight.shape  # N条光流
        re_blur = flow_warp(sharp, torch.cat([kernel[:, 0:1, :, :], kernel[:, N:N + 1, :, :]], dim=1)) * weight[:, 0:1, :, :]  # 用第0条光流warp出第0帧
        for i in range(1, N):
            flow = torch.cat([kernel[:, i:i + 1, :, :], kernel[:, i + N:i + N + 1, :, :]], dim=1)  # 第i条光流
            frame = flow_warp(sharp, flow)  # 第i帧
            frame = frame * weight[:, i:i + 1, :, :]  # 第i帧乘以权重
            re_blur = re_blur + frame  # N个帧累加
        return re_blur


class GradImage(nn.Module):
    """
    感谢开源项目 https://github.com/Anumol96/image_gradient
    """

    def __init__(self):
        super(GradImage, self).__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # x轴卷积核
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)  # x轴梯度卷积
        self.conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).cuda())

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # y轴卷积
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)  # y轴梯度卷积
        self.conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).cuda())

    def forward(self, img):
        x0 = img[:, 0:1, ...]  # b*3*h*w按RGB通道拆成b*1*h*w
        x1 = img[:, 1:2, ...]
        x2 = img[:, 2:3, ...]

        G0_x = self.conv1(x0)
        G1_x = self.conv1(x1)
        G2_x = self.conv1(x2)
        G_x = torch.cat([torch.sqrt(G0_x), torch.sqrt(G1_x), torch.sqrt(G2_x)], dim=1)  # RGB三通道梯度拼接到一起 b*3*h*w

        G0_y = self.conv2(x0)
        G1_y = self.conv2(x1)
        G2_y = self.conv2(x2)
        G_y = torch.cat([torch.sqrt(G0_y), torch.sqrt(G1_y), torch.sqrt(G2_y)], dim=1)  # RGB三通道梯度拼接到一起 b*3*h*w

        return G_x, G_y


class GradReBlur(nn.Module):
    def __init__(self):
        super(GradReBlur, self).__init__()
        self.re_blur = ReBlur()  # L2的ReBlur
        self.grad_image = GradImage()  # 求图像梯度

    def forward(self, blur, sharp, kernel, weight):
        grad_blur_x, grad_blur_y = self.grad_image(blur)
        grad_sharp_x, grad_sharp_y = self.grad_image(sharp)

        grad_blur_x = torch.where(torch.isnan(grad_blur_x), torch.full_like(grad_blur_x, 0), grad_blur_x)  # Nan替换为0
        grad_blur_y = torch.where(torch.isnan(grad_blur_y), torch.full_like(grad_blur_y, 0), grad_blur_y)
        grad_sharp_x = torch.where(torch.isnan(grad_sharp_x), torch.full_like(grad_sharp_x, 0), grad_sharp_x)
        grad_sharp_y = torch.where(torch.isnan(grad_sharp_y), torch.full_like(grad_sharp_y, 0), grad_sharp_y)

        re_blur_x = self.re_blur(grad_blur_x, grad_sharp_x, kernel, weight)
        re_blur_y = self.re_blur(grad_blur_y, grad_sharp_y, kernel, weight)
        return re_blur_x + re_blur_y


class RegularTerm(nn.Module):
    def __init__(self):
        super(RegularTerm, self).__init__()

    def forward(self, kernel):
        """
        正则化项
        让N个点到中心采样点的距离，的和，最小
        """
        _, N, _, _ = kernel.shape
        N = N // 2  # N条光流,N个点

        dis = torch.sqrt(kernel[:, 0:1, :, :] * kernel[:, 0:1, :, :] + kernel[:, N:N + 1, :, :] * kernel[:, N:N + 1, :, :])  # 第1个点到原点的距离
        for i in range(1, N):
            dis = dis + torch.sqrt(kernel[:, i:i + 1, :, :] * kernel[:, i:i + 1, :, :] + kernel[:, i + N:i + N + 1, :, :] * kernel[:, i + N:i + N + 1, :, :])  # 第i个点的距离
        return torch.mean(dis)
