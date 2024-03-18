"""
Implementation of CBAM: Convolutional Block Attention Module
for 3D data.

Some (if not most) of the code is adapted from the following:
    https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py
"""

import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module for 3D data.


    ----------
    in_channels : int
        Number of channels in the input data.
    reduction : int
        Reduction ratio for the number of channels in the
        intermediate layer.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        inner_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.SELU(),
            nn.Conv3d(inner_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatioTemporalAttentionModule(nn.Module):
    """
    Spatial and temporal  Attention Module for 3D data.

    Parameters
    ----------
    kernel_size : int
        Size of the convolutional kernel.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(
            2,
            1,
            (kernel_size, kernel_size, kernel_size),
            padding="same",
            bias=False,
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


class CBAM3D(nn.Module):
    """
    CBAM: Convolutional Block Attention Module for 3D data.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input data.
    reduction : int
        Reduction ratio for the number of channels in the
        intermediate layer.
    kernel_size : int
        Size of the convolutional kernel.
    """

    def __init__(self, in_channels, reduction=2, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatioTemporalAttentionModule(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

    def output_size(self, input_size):
        return input_size
