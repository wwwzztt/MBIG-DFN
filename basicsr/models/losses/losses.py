# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)


class MultiHeadPSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(MultiHeadPSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
        self.step = 0

    def forward(self, pred, target):
        assert len(pred.size()) == 5
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=2).unsqueeze(dim=2) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 5

        l2 = ((pred - target.unsqueeze(1)) ** 2).mean(dim=(2, 3, 4))  # N, H
        l2 = l2.min(dim=1)[0]
        return self.loss_weight * self.scale * torch.log(l2 + 1e-8).mean()


class MultiHeadL1Loss_FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', fft_weight=0.01):
        super(MultiHeadL1Loss_FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')
        self.reduction = reduction
        self.fft_weight = fft_weight  # Weight for FFT Loss

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, H, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        # L1 Loss
        pred = pred[-1]
        target_l1 = target.unsqueeze(1)  # (N, 1, C, H, W)
        l1_losses = torch.abs(pred - target_l1)  # (N, H, C, H, W)
        if self.reduction == 'mean':
            l1_losses = l1_losses.mean(dim=[2, 3, 4])  # (N, H)

        # FFT Loss
        target_fft = target.unsqueeze(1)  # (N, 1, C, H, W)
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))  # (N, H, C, H, W)
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)  # (N, H, C, H, W, 2)
        target_fft = torch.fft.fft2(target_fft, dim=(-2, -1))  # (N, 1, C, H, W)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)  # (N, 1, C, H, W, 2)
        fft_diff = torch.abs(pred_fft - target_fft)  # (N, H, C, H, W, 2)
        if self.reduction == 'mean':
            fft_losses = fft_diff.mean(dim=[2, 3, 4, 5])  # (N, H)

        # Combine losses
        sum_loss = l1_losses + self.fft_weight * fft_losses  # (N, H)

        # Find minimum loss across heads
        min_loss, min_head = sum_loss.min(dim=1)  # min_loss: (N,), min_head: (N,)

        return min_loss.mean()


class MIMOHeadL1Loss_FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', fft_weight=0.01):
        super(MIMOHeadL1Loss_FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')
        self.reduction = reduction
        self.fft_weight = fft_weight  # Weight for FFT Loss
        self.l1loss = L1Loss(reduction=reduction)
        self.fftloss = FFTLoss(reduction=reduction)

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, H, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        target2 = F.interpolate(target, scale_factor=0.5, mode='bilinear')
        target4 = F.interpolate(target, scale_factor=0.25, mode='bilinear')

        # 0
        pred0 = pred[0]
        l1_0 = self.l1loss(pred0, target4)
        fft_0 = self.fftloss(pred0, target4)

        # 1
        pred1 = pred[1]
        l1_1 = self.l1loss(pred1, target2)
        fft_1 = self.fftloss(pred1, target2)

        # 2
        pred2 = pred[2]
        # L1 Loss
        target_l1 = target.unsqueeze(1)  # (N, 1, C, H, W)
        l1_2 = torch.abs(pred2 - target_l1)  # (N, H, C, H, W)
        if self.reduction == 'mean':
            l1_2 = l1_2.mean(dim=[2, 3, 4])  # (N, H)

        # FFT Loss
        target_fft = target.unsqueeze(1)  # (N, 1, C, H, W)
        pred_fft = torch.fft.fft2(pred2, dim=(-2, -1))  # (N, H, C, H, W)
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)  # (N, H, C, H, W, 2)
        target_fft = torch.fft.fft2(target_fft, dim=(-2, -1))  # (N, 1, C, H, W)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)  # (N, 1, C, H, W, 2)
        fft_2 = torch.abs(pred_fft - target_fft)  # (N, H, C, H, W, 2)
        if self.reduction == 'mean':
            fft_2 = fft_2.mean(dim=[2, 3, 4, 5])  # (N, H)

        sum_loss = l1_2 + self.fft_weight * fft_2  # (N, H)
        min_loss, min_head = sum_loss.min(dim=1)  # min_loss: (N,), min_head: (N,)

        # Combine losses
        loss0 = l1_0 + self.fft_weight * fft_0
        loss1 = l1_1 + self.fft_weight * fft_1
        loss2 = min_loss
        sum_loss = loss2 + 0.1 * (loss0 + loss1)
        return sum_loss.mean()



