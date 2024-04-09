# -*- coding: utf-8 -*-
# @File  : GoldYolo_my.py
# @Author: yblir
# @Time  : 2024-04-09 21:29
# @Explain: 
# ======================================================================================================================
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# from yolov6.layers.common import SimConv
# from .transformer import onnx_AdaptiveAvgPool2d


__all__ = ['GoldYolo_my']


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


class Low_FAM(nn.Module):
    """浅层特征融合"""
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])

        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out
