# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 15:49
# @Author  : yblir
# @File    : ASCPA.py
# explain  : 
# =======================================================
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['ASCPA']

class ASCPA(nn.Module):
    def __init__(self, c1, inter_channels_rate=8, add=True):
        """
        Args:
            c1: input channel
            inter_channels_rate: channels inter rate
        """
        super(ASCPA, self).__init__()
        self.inter_channels = c1 // inter_channels_rate
        self.in_channels = c1
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1, bias=False)
        self.W = nn.Conv2d(self.inter_channels,
                           self.in_channels, 1, bias=False)

        self.down1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.down2 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.add = add
        self.vfc1 = nn.Linear(3, 16, bias=False)
        self.vfc2 = nn.Linear(16, 3, bias=False)
        self.soft = nn.Softmax(1)

    def forward(self, x):
        g_x = self.g(x)
        W_y = self.af(g_x, x, self.down1, self.down2)
        z = self.W(W_y)
        if self.add:
            z += x
        return z

    def af(self, g_x, x, down1, down2):
        batch_size = g_x.size(0)
        g_x = g_x.view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        # SPE kernel 1
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = x.view(batch_size, self.in_channels, -1)
        f1 = torch.matmul(theta_x, phi_x)

        # SPE kernel 2
        x2 = down1(x)
        theta_x = x2.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = x2.view(batch_size, self.in_channels, -1)
        f2 = torch.matmul(theta_x, phi_x)

        # SPE kernel 3
        x3 = down2(x)
        theta_x = x3.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = x3.view(batch_size, self.in_channels, -1)
        f3 = torch.matmul(theta_x, phi_x)

        # SCFM
        v1 = f1.mean(2).mean(1)
        v2 = f2.mean(2).mean(1)
        v3 = f3.mean(2).mean(1)
        V = torch.stack([v1, v2, v3], 1)
        V = self.soft(self.vfc2(self.vfc1(V)))
        f = f1 * V[:, 0].unsqueeze(1).unsqueeze(1) + \
            f2 * V[:, 1].unsqueeze(1).unsqueeze(1) + \
            f3 * V[:, 2].unsqueeze(1).unsqueeze(1)

        f_div_C = F.softmax(f, dim=-1)
        # dot attention
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        return y
