"""
Implements the TemporalBlock and TemporalEncoder classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    Encodes a sequence of frames into a latent representation, using a combination
    of local convolutions and global poolings.
    Based on Hu et. al. https://arxiv.org/pdf/2003.06409.pdf (2020).

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    height: int
        Height of the input frames.
    width: int
        Width of the input frames.
    out_channels: int
        Number of output channels.
    kernel_size_time: int
        Kernel size over the time dimension.
    kernel_size_space: int
        Kernel size over the space dimensions.
    """

    def __init__(
        self, in_channels, height, width, out_channels, kernel_size_time, kernel_size_space
    ):
        super(TemporalBlock, self).__init__()
        # Local context
        # First, a 1x1x1 conv to reduce the number of channels
        self.local_compressors = nn.ModuleList(
            [nn.Conv3d(in_channels, in_channels // 2, kernel_size=1) for _ in range(4)]
        )
        # Spatial motion
        c, kt, ks = in_channels // 2, kernel_size_time, kernel_size_space
        self.conv1 = nn.Conv3d(c, c, kernel_size=(1, ks, ks), padding="same")
        # Horizontal and vertical motions
        self.conv2 = nn.Conv3d(c, c, kernel_size=(kt, 1, ks), padding="same")
        self.conv3 = nn.Conv3d(c, c, kernel_size=(kt, ks, 1), padding="same")
        # Complete motion
        self.conv4 = nn.Conv3d(c, c, kernel_size=(kt, ks, ks), padding="same")
        # Global context
        self.pool1 = nn.AvgPool3d(kernel_size=(kt, height, width))
        self.pool2 = nn.AvgPool3d(kernel_size=(1, height // 2, width // 2))
        self.pool3 = nn.AvgPool3d(kernel_size=(1, height // 4, width // 4))
        # Channel reduction for the global context
        self.global_compressors = nn.ModuleList(
            [nn.Conv3d(in_channels, in_channels // 3, kernel_size=1) for _ in range(3)]
        )
        # Final convolution
        self.final_conv = nn.Conv3d(
            4 * (in_channels // 2) + 3 * (in_channels // 3), out_channels, kernel_size=1
        )
        # Batch normalization
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, C, T, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C', T, H, W).
        """
        # List of all tensors obtained across the local and global contexts
        middle_tensors = []
        # Local context
        # Apply the local compressors
        compressed = [compressor(x) for compressor in self.local_compressors]
        # Apply each type of motion convolution
        middle_tensors.append(self.conv1(compressed[0]))
        middle_tensors.append(self.conv2(compressed[1]))
        middle_tensors.append(self.conv3(compressed[2]))
        middle_tensors.append(self.conv4(compressed[3]))
        # Global context
        # Apply the global poolings
        pooled = [pool(x) for pool in [self.pool1, self.pool2, self.pool3]]
        # Apply the global compressors
        compressed = [
            compressor(pool) for compressor, pool in zip(self.global_compressors, pooled)
        ]
        # Upsample the global context tensors
        middle_tensors += [
            F.interpolate(z, size=(x.size(2), x.size(3), x.size(4))) for z in compressed
        ]
        # Concatenate all tensors
        middle_tensors = torch.cat(middle_tensors, dim=1)
        # Final convolution
        out = self.final_conv(middle_tensors)
        # Skip connection with the input tensor
        out += x
        # Batch normalization
        out = self.batch_norm(out)
        return out


class TemporalEncoder(nn.Module):
    """
    Assembles multiple TemporalBlock modules to encode a sequence of frames.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    height: int
        Height of the input frames.
    width: int
        Width of the input frames.
    kernel_size_time: int
        Kernel size over the time dimension.
    kernel_size_space: int
        Kernel size over the space dimensions.
    num_blocks: int
        Number of TemporalBlock modules to use.
    """

    def __init__(
        self,
        in_channels,
        height,
        width,
        kernel_size_time,
        kernel_size_space,
        num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    in_channels, height, width, in_channels, kernel_size_time, kernel_size_space
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, C, T, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C', T, H, W).
        """
        for block in self.blocks:
            x = block(x)
        return x
